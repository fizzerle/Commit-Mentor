import requests
from pprint import pprint
import urllib
import re
import logging
from pprint import pformat
import math
import json

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

token = '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'

def sendRequest(url,header = None, perPage = 1,page = 1, numberOfValues = None, additionalParamKey = None, additionalParamValue = None, getAll = False):
    params={}
    if getAll:
        perPage = 100
        page = 1
    if(header == None):
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'token {token}',
        }

    if numberOfValues:
        perPage = 100
    if(not '?' in url):
        params = {
            'per_page': perPage,
            'page': page,
        }
    
    if additionalParamKey:
        params[additionalParamKey] = additionalParamValue


    logging.debug("sending request: "+ url + str(params) + str(headers))

    r = requests.request('GET', url, params=params, headers=headers)
    if (r.status_code != 200):
        raise Exception(f'invalid github response: {r.content}')

    pagesCount = 0
    resp = r.json()
    last_page = r.links.get('last')

    if last_page:
        # extract the query string from the last page url
        qs = urllib.parse.urlparse(last_page['url']).query
        # extract the page number from the query string
        url = urllib.parse
        pagesCount = int(dict(urllib.parse.parse_qsl(qs))['page'])

    if numberOfValues or getAll:
        pagesNeeded = 0
        if numberOfValues:
            pagesNeeded = math.ceil(numberOfValues/perPage)
        count = 2
        while 'next' in r.links.keys() and (count <= pagesNeeded or getAll):
            logging.info("Fetching Page " +  str(count) + "/" + str(pagesCount))
            r = requests.request('GET', r.links['next']['url'], headers=headers)
            resp.extend(r.json())
            count += 1

    logging.trace("response: "+ pformat(resp,width = 180,compact = True))

    return resp

def getRepositories(stars,language,number):
    r = sendRequest("https://api.github.com/search/repositories",numberOfValues = number, additionalParamKey = 'q', additionalParamValue = 'stars:>{stars} language:{language}', perPage = 100)
    return r['items']


def commit_count(url, sha):
    """
    Return the number of commits to a project
    """
    resp = sendRequest(url,additionalParamKey='sha', additionalParamValue = sha)
    # check the resp count, just in case there are 0 commits
    commit_count = len(resp)
    last_page = resp.links.get('last')
    #pprint(resp.json())
    # if there are no more pages, the count must be 0 or 1
    if last_page:
        # extract the query string from the last page url
        qs = urllib.parse.urlparse(last_page['url']).query
        # extract the page number from the query string
        url = urllib.parse
        commit_count = int(dict(urllib.parse.parse_qsl(qs))['page'])
    return commit_count


def getCommitInRepoForUsers(repoUrl, branch, userIds,usernames = None, numberOfCommits = None):
    logging.info("Getting Commit Message In "+ repoUrl +" for Users " + str(userIds))
    if numberOfCommits:
        commits = sendRequest(repoUrl,additionalParamKey = 'sha', additionalParamValue = branch,numberOfValues = numberOfCommits)
    else:
        commits = sendRequest(repoUrl,additionalParamKey = 'sha', additionalParamValue = branch,getAll = True)

    logging.info(str(len(commits)) + ' Commits fetched')
    filterCommits = []
    for commit in commits:
        if commit['author']:
            commitAuthorId = commit['author']['id']
            noUser = True
            for userId in userIds:
                if(userId == commitAuthorId):
                    commitMessage = commit['commit']['message'].replace('\n',' ').replace('\r', '')
                    filterCommits.append({'sha': commit['sha'], 'message': commitMessage})
                    noUser = False
                
            if noUser:
                logging.trace(pformat(commit)) 
        elif commit['commit']:
            commiterName= commit['commit']['committer']['name']
            noUser = True
            for name in usernames:
                if(name == commiterName):
                    noUser = False
                    commitMessage = commit['commit']['message'].replace('\n',' ').replace('\r', '')
                    filterCommits.append({'sha': commit['sha'], 'message': commitMessage, 'html_url': commit['html_url'] })
            
            if noUser:
                logging.trace(pformat(commit))

    logging.info(str(len(filterCommits)) + ' Commits assigned')
    return filterCommits

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

def isAscii(s):
    return all(ord(c) < 128 for c in s)

def filterMessages(commits):
    f = open("filteredMessages.txt", "w", encoding='utf-8')
    stats = {'all': len(commits), 'empty': 0, 'merge':0, 'nonASCII': 0, 'rollback': 0, 'bot': 0, 'good': 0}
    remove = False
    for commit in commits:
        remove = False
        msg = commit['message']
        #check if commit is empty
        if not msg:
            remove = True
            stats['empty'] += 1
        if msg.startswith('merge'):
            remove = True
            stats['merge'] += 1
        if msg.startswith('rollback') or msg.startswith('revert'):
            remove = True
            stats['rollback'] += 1
        if not isAscii(msg):
            remove = True
            stats['nonASCII'] += 1
        #bot message
        if(re.search(r'^ignore	update	\'	.*	\.',msg) or
        re.search(r'^update(d)? (changelog|gitignore|readme( . md| file)?)( \.)?',msg) or
        re.search(r'^prepare version (v)?[ \d.]+',msg) or
        re.search(r'^bump (up )?version( number| code)?( to (v)?[ \d.]+( - snapshot)?)?( \.)?',msg) or
        re.search(r'^modify	(dockerfile|makefile)( \.)?',msg) or
        re.search(r'^update submodule(s)?( \.)?',msg)):
            remove = True
            stats['bot'] += 1
        if remove:
            f.write("---------\n"+"message: "+msg+"\nsha: "+commit['sha']+"\n")
        else:
            stats['good'] += 1

    logging.info(stats)
    f.close()

def removeNewlines(s):
    return s.replace("\n", " ").replace("\r", " ")

def resolveIssueIdsInCommitMessage(repo,commits):
    for idx,commit in enumerate(commits):
        ids = getIdsInMessage(commit['message'])
        if ids:
            commit['related'] = []
        logging.info("Resolving Issues of Commit #"+ str(idx))
        for id in ids:
            issueClean = {'id':id}
            issue = sendRequest(repo['issues_url'][:-9]+"/"+id)
            idsInIssue = []
            if(issue['body']):
                issueClean['body'] = removeNewlines(issue['body'])
                idsInIssue = getIdsInMessage(issue['body'])
            
            issueClean['title'] = removeNewlines(issue['title'])

            if idsInIssue:
                issueClean['mentionedInIssue'] = []
            for idInIssue in idsInIssue:
                issueCleanInner = {'id':idInIssue}
                issue = sendRequest(repo['issues_url'][:-9]+"/"+id)
                
                issueCleanInner['title'] = removeNewlines(issue['title'])
                if(issue['body']):
                    issueCleanInner['body'] = removeNewlines(issue['body'])
                
                issueClean['mentionedInIssue'].append(issueCleanInner)

            commit['related'].append(issueClean)


def analyzeRepo(repo,num_contributors,num_commits):
    f = open(repo['full_name']+".txt", "w", encoding='utf-8')
    contribtors = sendRequest(repo['contributors_url'],perPage = num_contributors)
    names = []
    ids = []
    logging.info("There are " + str(len(contribtors)) + " Contributors")
    for contributor in contribtors:
        name = sendRequest(contributor['url'])['name']
        if name:
            names.append(name)
            ids.append(contributor['id'])
        else:
            ids.append(contributor['id'])
    
    logging.info("Contributors are " + str(names))
    commits = getCommitInRepoForUsers(repo['commits_url'][:-6], repo['default_branch'], ids,names,num_commits)
    filterMessages(commits)
    resolveIssueIdsInCommitMessage(repo,commits)
    f.write(pformat(commits,width = 180,compact = True))
    f.close


repoName = "fizzerle/TISSFeedbacktool"
#repoName = "airbnb/lottie-android"
#repoName = "elastic/elasticsearch"

testRepo = {'full_name': repoName.replace('/','-'), 'contributors_url' :'https://api.github.com/repos/'+repoName+'/contributors', 'commits_url': 'https://api.github.com/repos/'+repoName+'/commits{/sha}', 'default_branch': 'master', 'issues_url': 'https://api.github.com/repos/'+repoName+'/issues{/number}'}
analyzeRepo(testRepo,10,100)

#repos = getBestJavaRepositories()
repos = []
selectedReposCount = 0
commitThreshold = 5000
for repo in repos:
    if(commit_count(repo['commits_url'][:-6], repo['default_branch']) > commitThreshold):
        logging.info("[x]" + repo['full_name'])
        contribtors = sendRequest(repo['contributors_url'])

        contributorIds = [contributor['id'] for contributor in contribtors]
        getCommitInRepoForUsers(repo['commits_url'][:-6], repo['default_branch'], contributorIds)
        selectedReposCount += 1
    else:
        logging.info("[ ]" + repo['full_name'])

logging.info("Repositories selected " + str(selectedReposCount)+"/"+str(len(repos))+" with Threshold " + str(commitThreshold))
