import requests
from pprint import pprint
import urllib
import re
import logging
from pprint import pformat

# Adding log level trace: https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/13638084#13638084
def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


addLoggingLevel('TRACE', logging.DEBUG - 5)

logging.basicConfig()

# By default the root logger is set to WARNING and all loggers you define
# inherit that value. Here we set the root logger to NOTSET. This logging
# level is automatically inherited by all existing and new sub-loggers
# that do not set a less verbose level.
logging.root.setLevel(logging.INFO)

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
    if(params == None and not '?' in url):
        params = {
            'per_page': 1,
            'page': 1,
        }
    logging.trace("sending request: "+ url + str(params) + str(headers))

    resp = requests.request('GET', url, params=params, headers=headers)
    if (resp.status_code != 200):
        raise Exception(f'invalid github response: {resp.content}')

    logging.trace("response: "+ pformat(resp.json()))
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
    logging.info("Getting Commit Message In Repo for Users " + str(userIds))
    params = {
        'sha': branch,
        'per_page': 10,
        'page': 1,
    }

    r = sendRequest(repoUrl,params)
    resp = r.json()
    while 'next' in r.links.keys():
        r=sendRequest(r.links['next']['url'])
        resp.extend(r.json())

    messages = []

    for commit in resp:
        commitAuthorId = commit['author']['id']
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
            #TODO filter messages are empty or automatically generated by a bot
            continue

def testGetCommitMessageInRepoForUser(token):
    messages = getCommitMessageInRepoForUsers("https://api.github.com/repos/elastic/elasticsearch/commits", "master", [14179713])
    filterMessages(messages)
    for msg in messages:
        ids = getIdsInMessage(msg)
        print(ids)
        for id in ids:
            issue = sendRequest("https://api.github.com/repos/elastic/elasticsearch/issues/"+id).json()
            idsInIssue = getIdsInMessage(issue['body'])
            for idInIssue in idsInIssue:
                issue = sendRequest("https://api.github.com/repos/elastic/elasticsearch/issues/"+idInIssue).json()

def testGetCommitMessageInRepoForUserSmallRepository(token):
    messages = getCommitMessageInRepoForUsers("https://api.github.com/repos/fizzerle/feedbacktool/commits", "master", [14179713])
    filterMessages(messages)
    for msg in messages:
        ids = getIdsInMessage(msg)
        for id in ids:
            issue = sendRequest("https://api.github.com/repos/fizzerle/feedbacktool/issues/"+id).json()
            idsInIssue = getIdsInMessage(issue['body'])
            for idInIssue in idsInIssue:
                issue = sendRequest("https://api.github.com/repos/fizzerle/feedbacktool/issues/"+idInIssue).json()


testGetCommitMessageInRepoForUserSmallRepository(token)

#repos = getBestJavaRepositories()
repos = []
sum = 0
for repo in repos:
    if(commit_count(repo['commits_url'][:-6], repo['default_branch'], token) > 5000):
        print("[x]" + repo['full_name'])
        contribtors = sendRequest(repo['contributors_url']).json()

        contributorIds = [contributor['id'] for contributor in contribtors]
        getCommitMessageInRepoForUsers(repo['commits_url'][:-6], repo['default_branch'], contributorIds)
        sum = sum + 1
    else:
        print("[ ]" + repo['full_name'])

logging.info("Number of Repositories selected that get processed further " + str(sum)+"/"+str(len(repos)))
