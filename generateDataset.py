import requests
from pprint import pprint
import urllib
import re

# search commits with certain text
token = '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'
owner = "MartinHeinz"
repo = "python-project-blueprint"
query_url = f"https://api.github.com/search/commits"
params = {
    'q': 'fix',
    'per_page': 1,
}
headers = {'Authorization': f'token {token}',
           "Accept": "application/vnd.github.cloak-preview+json"}
#r = requests.get(query_url, headers=headers, params=params)

def sendRequest(url,params = None,header = None):
    if(header == None):
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'token {token}',
        }
    if(params == None):
        params = {
            'per_page': 1,
            'page': 1,
        }
    resp = requests.request('GET', url, params=params, headers=headers)
    if (resp.status_code != 200):
        raise Exception(f'invalid github response: {resp.content}')

    return resp

def getBestJavaRepositories():
    params = {
        'q': 'stars:>10000 language:Java',
        'per_page': 6,
        'page': 1,
    }
    r = sendRequest("https://api.github.com/search/repositories",params = params).json()

    return r['items']


def commit_count(url, sha):
    """
    Return the number of commits to a project
    """
    params = {
        'sha': sha,
        'per_page': 1,
    }
    resp = sendRequest(url,params).json()
    # check the resp count, just in case there are 0 commits
    commit_count = len(resp)
    last_page = resp.links.get('last')
    #pprint(resp.json())
    # if there are no more pages, the count must be 0 or 1
    if last_page:
        # extract the query string from the last page url
        qs = urllib.parse.urlparse(last_page['url']).query
        # extract the page number from the query string
        commit_count = int(dict(urllib.parse.parse_qsl(qs))['page'])
    return commit_count


def getCommitMessageInRepoForUsers(repoUrl, branch, userIds):
    params = {
        'sha': branch,
        'per_page': 1,
        'page': 1,
    }

    #TODO 
    resp = sendRequest(repoUrl,params)
    while 'next' in resp.links.keys():
        res=sendRequest(res.links['next']['url']).json()
        resp.extend(res.json())
    pprint(resp)

    pprint(resp)
    messages = []

    for commit in resp:
        commitAuthorId = commit['author']['id']
        print(commitAuthorId)
        for userId in userIds:
            if(userId == commitAuthorId):
                messages.append(commit['commit']['message'])
    return messages

def getIdsInMessage(msg):
    # automatic linking of commits
    # https://docs.github.com/en/github/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests
    r = re.compile(r'#([0-9]+)|\bgh-([0-9]+)\b', flags=re.I)
    ids = []
    id = None
    for match in r.finditer(msg):
        if(match.group(1)):
            id = match.group(1)
        if(match.group(2)):
           id = match.group(2)
        ids.append(id)
    return ids

def filterMessages(messages):
    #check if list is empty
    if messages:
        for msg in messages:
            continue

def testGetCommitMessageInRepoForUser(token):
    messages = getCommitMessageInRepoForUsers("https://api.github.com/repos/elastic/elasticsearch/commits", "master", [40268737])
    filterMessages(messages)
    for msg in messages:
        print("----------------")
        print(msg)
        ids = getIdsInMessage(msg)
        print(ids)
        for id in ids:
            issue = sendRequest("https://api.github.com/repos/elastic/elasticsearch/issues/"+id).json()
            print(issue['body'])
            idsInIssue = getIdsInMessage(issue['body'])
            for idInIssue in idsInIssue:
                issue = sendRequest("https://api.github.com/repos/elastic/elasticsearch/issues/"+idInIssue).json()
                print(issue['body'])
        print("----------------")


testGetCommitMessageInRepoForUser(token)

#repos = getBestJavaRepositories()
repos = []
sum = 0
for repo in repos:
    if(commit_count(repo['commits_url'][:-6], repo['default_branch'], token) > 5000):
        print("--> " + repo['full_name'])
        contribtors = sendRequest(repo['contributors_url']).json()

        contributorIds = [contributor['id'] for contributor in contribtors]
        getCommitMessageInRepoForUsers(repo['commits_url'][:-6], repo['default_branch'], contributorIds)
        sum = sum + 1
    else:
        print(repo['full_name'])

print(str(sum)+"/"+str(len(repos)))
