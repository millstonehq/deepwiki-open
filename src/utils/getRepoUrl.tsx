import RepoInfo from "@/types/repoinfo";

export default function getRepoUrl(repoInfo: RepoInfo): string {
  console.log('getRepoUrl', repoInfo);
  if (repoInfo.type === 'local' && repoInfo.localPath) {
    return repoInfo.localPath;
  } else {
    if(repoInfo.repoUrl) {
      return repoInfo.repoUrl;
    } else {
      if(repoInfo.owner && repoInfo.repo) {
        // Construct proper URL based on repo type
        const baseUrls: Record<string, string> = {
          'github': 'https://github.com',
          'gitlab': 'https://gitlab.com',
          'bitbucket': 'https://bitbucket.org',
        };
        const baseUrl = baseUrls[repoInfo.type || 'github'] || 'https://github.com';
        return `${baseUrl}/${repoInfo.owner}/${repoInfo.repo}`;
      }
      return '';
    }
  }
};