"""
Server-side wiki generation orchestrator.

Handles the complete flow: clone repo -> generate structure -> generate pages -> cache results.
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

from api.job_manager import JobManager, JobStatus, WikiGenerationJob
from api.data_pipeline import download_repo
from api.vertex_client import VertexAIClient
from api.models import (
    WikiPage,
    WikiStructureModel,
    WikiSection,
    RepoInfo,
    WikiCacheRequest,
)

logger = logging.getLogger(__name__)

# Language mapping for prompts
LANGUAGE_NAMES = {
    'en': 'English',
    'ja': 'Japanese (日本語)',
    'zh': 'Mandarin Chinese (中文)',
    'zh-tw': 'Traditional Chinese (繁體中文)',
    'es': 'Spanish (Español)',
    'kr': 'Korean (한국어)',
    'vi': 'Vietnamese (Tiếng Việt)',
    'pt-br': 'Brazilian Portuguese (Português Brasileiro)',
    'fr': 'Français (French)',
    'ru': 'Русский (Russian)',
}


def get_adalflow_default_root_path() -> str:
    """Get the adalflow root path for storage."""
    return os.path.expanduser(os.path.join("~", ".adalflow"))


def build_structure_prompt(
    owner: str,
    repo: str,
    file_tree: str,
    readme: str,
    language: str,
    comprehensive: bool
) -> str:
    """Build the wiki structure generation prompt."""
    language_name = LANGUAGE_NAMES.get(language, 'English')

    comprehensive_sections = """
Create a structured wiki with the following main sections:
- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality)
- Data Management/Flow: If applicable, how data is stored, processed, accessed, and managed (e.g., database schema, data pipelines, state management).
- Frontend Components (UI elements, if applicable.)
- Backend Systems (server-side components)
- Model Integration (AI model connections)
- Deployment/Infrastructure (how to deploy, what's the infrastructure like)
- Extensibility and Customization: If the project architecture supports it, explain how to extend or customize its functionality (e.g., plugins, theming, custom modules, hooks).

Each section should contain relevant pages. For example, the "Frontend Components" section might include pages for "Home Page", "Repository Wiki Page", "Ask Component", etc.

Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <sections>
    <section id="section-1">
      <title>[Section title]</title>
      <pages>
        <page_ref>page-1</page_ref>
        <page_ref>page-2</page_ref>
      </pages>
      <subsections>
        <section_ref>section-2</section_ref>
      </subsections>
    </section>
    <!-- More sections as needed -->
  </sections>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>page-2</related>
        <!-- More related page IDs as needed -->
      </related_pages>
      <parent_section>section-1</parent_section>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>
"""

    concise_structure = """
Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>page-2</related>
        <!-- More related page IDs as needed -->
      </related_pages>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>
"""

    page_count = '8-12' if comprehensive else '4-6'
    wiki_type = 'comprehensive' if comprehensive else 'concise'

    return f"""Analyze this GitHub repository {owner}/{repo} and create a wiki structure for it.

1. The complete file tree of the project:
<file_tree>
{file_tree}
</file_tree>

2. The README file of the project:
<readme>
{readme}
</readme>

I want to create a wiki for this repository. Determine the most logical structure for a wiki based on the repository's content.

IMPORTANT: The wiki content will be generated in {language_name} language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as:
- Architecture overviews
- Data flow descriptions
- Component relationships
- Process workflows
- State machines
- Class hierarchies

{comprehensive_sections if comprehensive else concise_structure}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid XML structure specified above
- DO NOT wrap the XML in markdown code blocks (no ``` or ```xml)
- DO NOT include any explanation text before or after the XML
- Ensure the XML is properly formatted and valid
- Start directly with <wiki_structure> and end with </wiki_structure>

IMPORTANT:
1. Create {page_count} pages that would make a {wiki_type} wiki for this repository
2. Each page should focus on a specific aspect of the codebase (e.g., architecture, key features, setup)
3. The relevant_files should be actual files from the repository that would be used to generate that page
4. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters"""


def build_page_prompt(
    page_title: str,
    file_paths: List[str],
    repo_url: str,
    language: str
) -> str:
    """Build the wiki page generation prompt."""
    language_name = LANGUAGE_NAMES.get(language, 'English')

    # Generate file URLs (simplified for server-side)
    file_list = '\n'.join([f"- {path}" for path in file_paths])

    return f"""You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format about a specific feature, system, or module within a given software project.

You will be given:
1. The "[WIKI_PAGE_TOPIC]" for the page you need to create.
2. A list of "[RELEVANT_SOURCE_FILES]" from the project that you MUST use as the sole basis for the content. You have access to the full content of these files. You MUST use AT LEAST 5 relevant source files for comprehensive coverage - if fewer are provided, search for additional related files in the codebase.

CRITICAL STARTING INSTRUCTION:
The very first thing on the page MUST be a `<details>` block listing ALL the `[RELEVANT_SOURCE_FILES]` you used to generate the content. There MUST be AT LEAST 5 source files listed - if fewer were provided, you MUST find additional related files to include.
Format it exactly like this:
<details>
<summary>Relevant source files</summary>

Remember, do not provide any acknowledgements, disclaimers, apologies, or any other preface before the `<details>` block. JUST START with the `<details>` block.
The following files were used as context for generating this wiki page:

{file_list}
<!-- Add additional relevant files if fewer than 5 were provided -->
</details>

Immediately after the `<details>` block, the main title of the page should be a H1 Markdown heading: `# {page_title}`.

Based ONLY on the content of the `[RELEVANT_SOURCE_FILES]`:

1.  **Introduction:** Start with a concise introduction (1-2 paragraphs) explaining the purpose, scope, and high-level overview of "{page_title}" within the context of the overall project. If relevant, and if information is available in the provided files, link to other potential wiki pages using the format `[Link Text](#page-anchor-or-id)`.

2.  **Detailed Sections:** Break down "{page_title}" into logical sections using H2 (`##`) and H3 (`###`) Markdown headings. For each section:
    *   Explain the architecture, components, data flow, or logic relevant to the section's focus, as evidenced in the source files.
    *   Identify key functions, classes, data structures, API endpoints, or configuration elements pertinent to that section.

3.  **Mermaid Diagrams:**
    *   EXTENSIVELY use Mermaid diagrams (e.g., `flowchart TD`, `sequenceDiagram`, `classDiagram`, `erDiagram`, `graph TD`) to visually represent architectures, flows, relationships, and schemas found in the source files.
    *   Ensure diagrams are accurate and directly derived from information in the `[RELEVANT_SOURCE_FILES]`.
    *   Provide a brief explanation before or after each diagram to give context.
    *   CRITICAL: All diagrams MUST follow strict vertical orientation:
       - Use "graph TD" (top-down) directive for flow diagrams
       - NEVER use "graph LR" (left-right)
       - Maximum node width should be 3-4 words
       - For sequence diagrams:
         - Start with "sequenceDiagram" directive on its own line
         - Define ALL participants at the beginning using "participant" keyword
         - Use descriptive but concise participant names, or use aliases: "participant A as Alice"
         - Use the correct Mermaid arrow syntax:
           - ->> solid line with arrowhead (most common for requests/calls)
           - -->> dotted line with arrowhead (most common for responses/returns)
           - ->x solid line with X at end (failed/error message)
           - -) solid line with open arrow (async message, fire-and-forget)
         - NEVER use flowchart-style labels like A--|label|-->B. Always use a colon for labels: A->>B: My Label

4.  **Tables:**
    *   Use Markdown tables to summarize information such as:
        *   Key features or components and their descriptions.
        *   API endpoint parameters, types, and descriptions.
        *   Configuration options, their types, and default values.
        *   Data model fields, types, constraints, and descriptions.

5.  **Code Snippets (ENTIRELY OPTIONAL):**
    *   Include short, relevant code snippets (e.g., Python, Java, JavaScript, SQL, JSON, YAML) directly from the `[RELEVANT_SOURCE_FILES]` to illustrate key implementation details, data structures, or configurations.
    *   Ensure snippets are well-formatted within Markdown code blocks with appropriate language identifiers.

6.  **Source Citations (EXTREMELY IMPORTANT):**
    *   For EVERY piece of significant information, explanation, diagram, table entry, or code snippet, you MUST cite the specific source file(s) and relevant line numbers from which the information was derived.
    *   Place citations at the end of the paragraph, under the diagram/table, or after the code snippet.
    *   Use the exact format: `Sources: [filename.ext:start_line-end_line]()` for a range, or `Sources: [filename.ext:line_number]()` for a single line.
    *   IMPORTANT: You MUST cite AT LEAST 5 different source files throughout the wiki page to ensure comprehensive coverage.

7.  **Technical Accuracy:** All information must be derived SOLELY from the `[RELEVANT_SOURCE_FILES]`. Do not infer, invent, or use external knowledge about similar systems or common practices unless it's directly supported by the provided code.

8.  **Clarity and Conciseness:** Use clear, professional, and concise technical language suitable for other developers working on or learning about the project.

9.  **Conclusion/Summary:** End with a brief summary paragraph if appropriate for "{page_title}", reiterating the key aspects covered and their significance within the project.

IMPORTANT: Generate the content in {language_name} language.

Remember:
- Ground every claim in the provided source files.
- Prioritize accuracy and direct representation of the code's functionality and structure.
- Structure the document logically for easy understanding by other developers."""


def parse_structure_xml(response_text: str) -> WikiStructureModel:
    """Parse XML response into WikiStructureModel."""
    # Extract XML from response
    xml_match = re.search(r'<wiki_structure>[\s\S]*?</wiki_structure>', response_text)
    if not xml_match:
        logger.error(f"No valid XML found in response: {response_text[:500]}...")
        raise ValueError("No valid wiki_structure XML found in LLM response")

    xml_text = xml_match.group(0)

    # Clean any null characters
    xml_text = xml_text.replace('\x00', '')

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as e:
        logger.error(f"XML parse error: {e}")
        raise ValueError(f"Invalid XML in response: {e}")

    # Parse basic structure
    title = root.findtext('title', 'Wiki')
    description = root.findtext('description', '')

    # Parse sections if present
    sections: List[WikiSection] = []
    root_sections: List[str] = []
    sections_el = root.find('sections')
    if sections_el is not None:
        for section_el in sections_el.findall('section'):
            section_id = section_el.get('id', f'section-{len(sections)+1}')
            section_title = section_el.findtext('title', '')
            page_refs = [pr.text for pr in section_el.findall('.//page_ref') if pr.text]
            subsection_refs = [sr.text for sr in section_el.findall('.//section_ref') if sr.text]

            sections.append(WikiSection(
                id=section_id,
                title=section_title,
                pages=page_refs,
                subsections=subsection_refs if subsection_refs else None
            ))
            root_sections.append(section_id)

    # Parse pages
    pages: List[WikiPage] = []
    for page_el in root.findall('.//page'):
        page_id = page_el.get('id', f'page-{len(pages)+1}')
        page_title = page_el.findtext('title', '')
        page_desc = page_el.findtext('description', '')
        importance = page_el.findtext('importance', 'medium')
        file_paths = [fp.text for fp in page_el.findall('.//file_path') if fp.text]
        related = [r.text for r in page_el.findall('.//related') if r.text]

        pages.append(WikiPage(
            id=page_id,
            title=page_title,
            content='',  # Will be generated later
            filePaths=file_paths,
            importance=importance,
            relatedPages=related
        ))

    logger.info(f"Parsed wiki structure: {title} with {len(pages)} pages")

    return WikiStructureModel(
        id='wiki',
        title=title,
        description=description,
        pages=pages,
        sections=sections if sections else None,
        rootSections=root_sections if root_sections else None
    )


class WikiGenerator:
    """Server-side wiki generation orchestrator."""

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self._client: Optional[VertexAIClient] = None

    def _get_client(self) -> VertexAIClient:
        """Get or create Vertex AI client."""
        if self._client is None:
            self._client = VertexAIClient()
        return self._client

    async def generate_wiki(self, job: WikiGenerationJob) -> None:
        """Main orchestration method - runs in background task."""
        try:
            # Phase 1: Clone repository
            await self.job_manager.update_job(
                job.id,
                status=JobStatus.CLONING,
                progress=5,
                message="Cloning repository..."
            )

            repo_path = await self._clone_repository(job)

            # Phase 2: Analyze repository structure
            await self.job_manager.update_job(
                job.id,
                status=JobStatus.ANALYZING,
                progress=10,
                message="Analyzing repository structure..."
            )

            file_tree, readme = self._get_repo_structure(repo_path)

            # Phase 3: Generate wiki structure
            await self.job_manager.update_job(
                job.id,
                status=JobStatus.GENERATING_STRUCTURE,
                progress=15,
                message="Generating wiki structure..."
            )

            wiki_structure = await self._generate_structure(job, file_tree, readme)
            pages_total = len(wiki_structure.pages)

            await self.job_manager.update_job(
                job.id,
                pages_total=pages_total,
                message=f"Found {pages_total} pages to generate"
            )

            # Phase 4: Generate pages
            await self.job_manager.update_job(
                job.id,
                status=JobStatus.GENERATING_PAGES,
                progress=20,
                message="Generating wiki pages..."
            )

            generated_pages = await self._generate_all_pages(job, wiki_structure, repo_path)

            # Phase 5: Cache and format results
            await self.job_manager.update_job(
                job.id,
                status=JobStatus.CACHING,
                progress=95,
                message="Caching results..."
            )

            result = await self._finalize_result(job, wiki_structure, generated_pages)

            # Complete
            await self.job_manager.update_job(
                job.id,
                status=JobStatus.COMPLETE,
                progress=100,
                message="Wiki generation complete",
                result=result
            )

            logger.info(f"Job {job.id} completed successfully")

        except Exception as e:
            logger.error(f"Wiki generation failed for job {job.id}: {e}", exc_info=True)
            await self.job_manager.update_job(
                job.id,
                status=JobStatus.ERROR,
                error=str(e),
                message=f"Generation failed: {str(e)}"
            )

    async def _clone_repository(self, job: WikiGenerationJob) -> str:
        """Clone the repository and return local path."""
        # Extract owner/repo from URL
        url_parts = job.repo_url.rstrip('/').split('/')
        owner = url_parts[-2]
        repo = url_parts[-1].replace('.git', '')

        # Determine repo type
        if 'gitlab' in job.repo_url.lower():
            repo_type = 'gitlab'
        elif 'bitbucket' in job.repo_url.lower():
            repo_type = 'bitbucket'
        else:
            repo_type = 'github'

        # Set up paths
        root_path = get_adalflow_default_root_path()
        repo_name = f"{owner}_{repo}"
        save_repo_dir = os.path.join(root_path, "repos", repo_name)

        # Run clone in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            download_repo,
            job.repo_url,
            save_repo_dir,
            repo_type,
            job.token
        )

        logger.info(f"Repository cloned to {save_repo_dir}")
        return save_repo_dir

    def _get_repo_structure(self, repo_path: str) -> Tuple[str, str]:
        """Get file tree and README content from cloned repository."""
        file_tree_lines: List[str] = []
        readme_content = ""

        # Default excluded directories
        excluded_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            'dist', 'build', '.next', 'coverage', '.tox', '.pytest_cache'
        }

        for root, dirs, files in os.walk(repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]

            for file in files:
                if file.startswith('.') or file == '__init__.py':
                    continue

                rel_dir = os.path.relpath(root, repo_path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file)

                # Find README
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README: {e}")

        file_tree = '\n'.join(sorted(file_tree_lines))
        logger.info(f"Found {len(file_tree_lines)} files in repository")
        return file_tree, readme_content

    async def _generate_structure(
        self,
        job: WikiGenerationJob,
        file_tree: str,
        readme: str
    ) -> WikiStructureModel:
        """Generate wiki structure using LLM."""
        # Extract owner/repo from URL
        url_parts = job.repo_url.rstrip('/').split('/')
        owner = url_parts[-2]
        repo = url_parts[-1].replace('.git', '')

        # Build prompt
        prompt = build_structure_prompt(
            owner, repo, file_tree, readme,
            job.language, job.comprehensive
        )

        # Call LLM
        response_text = await self._call_llm(job, prompt)

        # Parse XML response
        return parse_structure_xml(response_text)

    async def _generate_all_pages(
        self,
        job: WikiGenerationJob,
        wiki_structure: WikiStructureModel,
        repo_path: str
    ) -> Dict[str, WikiPage]:
        """Generate all wiki pages sequentially."""
        generated_pages: Dict[str, WikiPage] = {}
        total_pages = len(wiki_structure.pages)

        for i, page in enumerate(wiki_structure.pages):
            progress = 20 + int((i / total_pages) * 70)

            await self.job_manager.update_job(
                job.id,
                current_page=page.title,
                pages_completed=i,
                progress=progress,
                message=f"Generating page {i+1}/{total_pages}: {page.title}"
            )

            try:
                content = await self._generate_page_content(job, page, repo_path)

                # Create updated page with content
                updated_page = WikiPage(
                    id=page.id,
                    title=page.title,
                    content=content,
                    filePaths=page.filePaths,
                    importance=page.importance,
                    relatedPages=page.relatedPages
                )
                generated_pages[page.id] = updated_page
                logger.info(f"Generated page: {page.title}")

            except Exception as e:
                logger.error(f"Failed to generate page {page.title}: {e}")
                # Create page with error content
                generated_pages[page.id] = WikiPage(
                    id=page.id,
                    title=page.title,
                    content=f"# {page.title}\n\nError generating content: {str(e)}",
                    filePaths=page.filePaths,
                    importance=page.importance,
                    relatedPages=page.relatedPages
                )

        return generated_pages

    async def _generate_page_content(
        self,
        job: WikiGenerationJob,
        page: WikiPage,
        repo_path: str
    ) -> str:
        """Generate content for a single wiki page."""
        # Build prompt
        prompt = build_page_prompt(
            page.title,
            page.filePaths,
            job.repo_url,
            job.language
        )

        # Call LLM
        content = await self._call_llm(job, prompt)

        # Clean up markdown delimiters if present
        content = re.sub(r'^```markdown\s*', '', content)
        content = re.sub(r'```\s*$', '', content)

        return content

    async def _call_llm(self, job: WikiGenerationJob, prompt: str) -> str:
        """Call Vertex AI and return response text."""
        client = self._get_client()

        api_kwargs = client.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs={
                "model": job.model,
                "temperature": 0.7,
                "max_output_tokens": 8192
            }
        )

        response = await client.acall(api_kwargs=api_kwargs)

        # Extract text from response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return str(response)

    async def _finalize_result(
        self,
        job: WikiGenerationJob,
        wiki_structure: WikiStructureModel,
        generated_pages: Dict[str, WikiPage]
    ) -> Any:
        """Cache results and format final output."""
        # Update structure with generated content
        wiki_structure.pages = list(generated_pages.values())

        # Build RepoInfo
        url_parts = job.repo_url.rstrip('/').split('/')
        owner = url_parts[-2]
        repo = url_parts[-1].replace('.git', '')

        if 'gitlab' in job.repo_url.lower():
            repo_type = 'gitlab'
        elif 'bitbucket' in job.repo_url.lower():
            repo_type = 'bitbucket'
        else:
            repo_type = 'github'

        repo_info = RepoInfo(
            owner=owner,
            repo=repo,
            type=repo_type,
            repoUrl=job.repo_url
        )

        # Save to cache
        try:
            from api.api import save_wiki_cache
            cache_request = WikiCacheRequest(
                repo=repo_info,
                language=job.language,
                wiki_structure=wiki_structure,
                generated_pages=generated_pages,
                provider=job.provider,
                model=job.model
            )
            await save_wiki_cache(cache_request)
            logger.info(f"Saved wiki cache for {owner}/{repo}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        # Format output based on requested format
        if job.format == "markdown":
            return self._generate_markdown_output(wiki_structure, job.repo_url)
        else:
            return {
                "wiki_structure": wiki_structure.model_dump(),
                "generated_pages": {k: v.model_dump() for k, v in generated_pages.items()},
                "repo": repo_info.model_dump(),
                "provider": job.provider,
                "model": job.model
            }

    def _generate_markdown_output(self, wiki_structure: WikiStructureModel, repo_url: str) -> str:
        """Generate markdown export of wiki."""
        lines = [
            f"# {wiki_structure.title}",
            "",
            f"> {wiki_structure.description}",
            "",
            f"Repository: {repo_url}",
            "",
            "---",
            "",
            "## Table of Contents",
            ""
        ]

        # Add TOC
        for page in wiki_structure.pages:
            anchor = page.title.lower().replace(' ', '-')
            lines.append(f"- [{page.title}](#{anchor})")

        lines.extend(["", "---", ""])

        # Add pages
        for page in wiki_structure.pages:
            lines.append(page.content)
            lines.extend(["", "---", ""])

        return '\n'.join(lines)
