import requests
from pprint import pprint
import urllib
import re
import logging
from pprint import pformat
import math
import pickle
import json
from unidiff import PatchSet
from tree_sitter import Language, Parser
import pygit2
from pygit2 import clone_repository
import shutil
import os.path
from os import path
from unidiff import PatchSet

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
        raise Exception(f'invalid github response: {r.status_code} {r.content}')

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

# Returns commits of a repository for a group of users
def getCommitInRepoForUsers(repoUrl, branch, userIds,usernames = None, numberOfCommits = None):
    logging.info("Getting Commit Message In "+ repoUrl +" for Users " + str(userIds))
    if numberOfCommits:
        commits = sendRequest(repoUrl,additionalParamKey = 'sha', additionalParamValue = branch,numberOfValues = numberOfCommits)
    else:
        commits = sendRequest(repoUrl,getAll = True)

    logging.info(str(len(commits)) + ' Commits fetched')
    filterCommits = []
    #filterCommits = [{'sha': commit['sha'], 'message': commit['commit']['message'].replace('\n',' ').replace('\r', ''), 'html_url': commit['html_url'] } for commit in commits]
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
    """
    Return ONLY commits that are not bot generated
    """
    f = open("filteredMessages.txt", "w", encoding='utf-8')
    stats = {'all': len(commits), 'empty': 0, 'merge':0, 'nonASCII': 0, 'rollback': 0, 'bot': 0, 'good': 0}
    remove = False
    filteredCommits = []
    for commit in commits:
        remove = False
        msg = commit['message']
        #check if commit is empty
        if not msg:
            remove = True
            stats['empty'] += 1
        if re.match('^merge', msg, re.I):
            remove = True
            stats['merge'] += 1
        if re.match('^rollback|^revert', msg, re.I):
            remove = True
            stats['rollback'] += 1
        if not isAscii(msg):
            remove = True
            stats['nonASCII'] += 1
        #bot message
        if(re.search(r'^ignore	update	\'	.*	\.',msg, re.I) or
        re.search(r'^update(d)? (changelog|gitignore|readme( . md| file)?)( \.)?',msg, re.I) or
        re.search(r'^prepare version (v)?[ \d.]+',msg, re.I) or
        re.search(r'^bump ',msg, re.I) or
        re.search(r'^modify	(dockerfile|makefile)( \.)?',msg, re.I) or
        re.search(r'^update submodule(s)?( \.)?',msg, re.I) or
        re.search(r'bot',msg, re.I)):

            remove = True
            stats['bot'] += 1
        if remove:
            f.write("---------\n"+"message: "+msg+"\nsha: "+commit['sha']+"\n")
        else:
            filteredCommits.append(commit)
            stats['good'] += 1

    logging.info(stats)
    f.close()
    return filteredCommits

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

def countHowManyCommitsToIssues(commits):
    count = 0
    for commit in commits:
        ids = getIdsInMessage(commit['message'])
        if len(ids) != 0:
            count += 1
    logging.info("From "+ str(commits)+ "commits, "+str(count)+" have a linked issue ")

def letters_same(a,b):
    if(len(a) > len(b)):
        return sum ( a[i] == b[i] for i in range(len(b)) )
    else:
        return sum ( a[i] == b[i] for i in range(len(a)) )

def loadCommitsAndAnalzeHowManyMessagesAreSimilarForRepo(testRepo):
    commits = pickle.load(open(testRepo['full_name']+'_pickle'+".txt", "rb"))
    simmilar = []
    for idx,commit in enumerate(commits):
        if len(commit['message']) > 50:
            continue
        for j in range(idx+1,len(commits)):
                same = letters_same(commit['message'],commits[j]['message'])
                if(same > 15):
                    simmilar.append([commit['message'],commits[j]['message']])

    f = open(testRepo['full_name']+".txt", "w", encoding='utf-8')
    f.write(pformat(simmilar,width = 180))
    f.close

# returns the filtered commits of the most active contributors of a repository
def analyzeRepo(repo,num_contributors,num_commits = None):
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
    filteredCommits = filterMessages(commits)
    #resolveIssueIdsInCommitMessage(repo,filteredCommits)
    return filteredCommits

def findReposWithMoreCommitsThan(repos,commitThreshold = 5000):
    selectedReposCount = 0
    filterdRepos = []
    for repo in repos:
        if(commit_count(repo['commits_url'][:-6], repo['default_branch']) > commitThreshold):
            logging.info("[x]" + repo['full_name'])
            filterdRepos.append(repo)
            selectedReposCount += 1
        else:
            repos.remove(repo['full_name'])
            logging.info("[ ]" + repo['full_name'])
    logging.info("Repositories selected " + str(selectedReposCount)+"/"+str(len(repos))+" with Threshold " + str(commitThreshold))
    return filterdRepos

# Returns all commits that contain program elements in commit messages

# 1. Get the diffs of the commits
# 2. Filter out Text Files
# 3. Filter out comments
# 4. split diff by whiteSpace
# 5. Build bag of words of all splitted tokens
# 6. Split commit message by whiteSpace
# 7. Find intersection of the two sets
#https://github.com/octocat/hello-world/commit/553c2077f0edc3d5dc5d17262f6aa498e69d6f8e.diff

def findCommitsThatContainProgramElements(repo,commits):
    filterdCommits = []
    for idx,commit in enumerate(commits):
        logging.info("Fetching Diff " + str(idx) + "/" + str(len(commits)))
        diff = urllib.request.urlopen("https://github.com/" + repo['full_name'].replace('-','/') + "/commit/" + commit['sha']+'.diff')
        encoding = diff.headers.get_charsets()[0]
        patches = PatchSet(diff, encoding=encoding)

        tokens = re.findall('[A-Za-z]+',str(patches))

#        for idx, patch in enumerate(patches):
 #           for idy, hunk in enumerate(patch):
  #              for idz, line in enumerate(hunk):
   #                 if idx == 0 and idy == 1:
    #                    
        commit['diff'] = str(patches)
        # TODO: Filter text files?
        # TODO: Filter comment lines?

        splittedMessage = re.findall('[A-Za-z]+',commit['message'])
        common_elements = list(set(tokens).intersection(set(splittedMessage)))
        if(len(common_elements) > 0):
            commit['common'] = common_elements
            filterdCommits.append(commit)

    logging.info("There are " + str(len(filterdCommits)) + "/" + str(len(commits)) + " commits with program elements" )
    return filterdCommits

def writeCommitsToPickleFile(filename, commits):
    pickleFile = open(filename, "wb")
    pickle.dump(commits,pickleFile)
    pickleFile.close

def writeCommitsToFile(filename, commits):
    allCommitsFile = open(filename, "w", encoding='utf-8')
    allCommitsFile.write(pformat(commits,width = 180))
    allCommitsFile.close

def generateSyntaxtTree(sourceCode):
    Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',
    # Include one or more languages
    [
        './tree-sitter-java'
    ]
    )
    JAVA_LANGUAGE = Language('build/my-languages.so', 'java')

    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    tree = parser.parse(sourceCode)
    captures = getAllFunctions(JAVA_LANGUAGE, tree)
    splittedSource = sourceCode.decode("utf-8").split('\n')

    for cap in captures:
        line = splittedSource[cap[0].start_point[0]]
        #print(line)
        #get function name
        #print(line[cap[0].start_point[1]:cap[0].end_point[1]])
    return tree

def node_text(source, node):
    return bytes(source, "utf8")[node.start_byte:node.end_byte].decode("utf-8")

def findNodeTypeAtPosition(tree,line,start,end):
    #logging.info("Finding Node Type")
    currentNode = tree.root_node
    parent = tree.root_node
    while currentNode.start_point[0] != line and currentNode.end_point[0] != line:
        if len(currentNode.children) == 0:
            break
        for childNode in currentNode.children:
            if childNode.start_point[0] <= line and childNode.end_point[0] >= line:
                parent = currentNode
                currentNode = childNode
                break

    #print("Parent: " + parent.type)
    #print("Child: " + currentNode.type)
    #print(sourceCode[parent.start_point[0]:parent.end_point[0]])
    #print(currentNode.type)
    #print(parent.type)
    #TODO: use other type. Parent is not always the best
    return currentNode.type

# https://medium.com/codist-ai/introducing-tree-hugger-source-code-mining-for-human-b5fcd31bef55
# https://github.blog/2020-08-04-codegen-semantics-improved-language-support-system/
# https://github.com/autosoft-dev/tree-hugger
def getAllFunctions(language,tree):
        query = language.query(
            """
            (method_declaration
  name: (identifier) @function.method)
            """
        )
        return query.captures(tree.root_node)

def test():
    #shutil.rmtree('./RxJava')
    #repoName = "fizzerle/TISSFeedbacktool"
    #repoName = "airbnb/lottie-android"
    #repoName = "elastic/elasticsearch"
    repoName = "ReactiveX/RxJava"

    testRepo = {'full_name': repoName.replace('/','-'), 'contributors_url' :'https://api.github.com/repos/'+repoName+'/contributors', 'commits_url': 'https://api.github.com/repos/'+repoName+'/commits{/sha}', 'default_branch': '3.x', 'issues_url': 'https://api.github.com/repos/'+repoName+'/issues{/number}'}
    #commits = [{'sha': '17e71ab4e6dc432ac536bf5bf193055b32b3fb17', 'message': 'adsfadf adfadfdf adfdfdd hallo assertValue assertErrorMessage asdfsdfdf'}]
    filtered_commits = analyzeRepo(testRepo,2,20)
    commitsWithProgramElemet = findCommitsThatContainProgramElements(testRepo,filtered_commits)
    commits = analyzeRepo(testRepo,10,500)

    #for commit in commitsWithProgramElemet:
    #    generateSyntaxtTree(commit)

    writeCommitsToFile(testRepo['full_name']+".txt",commitsWithProgramElemet)
    writeCommitsToPickleFile(testRepo['full_name']+'_pickle'+".txt",commitsWithProgramElemet)

    #repos = getBestJavaRepositories()
    #filteredRepos = findReposWithMoreCommitsThan(repos,5000)


def enrichCommitWithConcreteSyntaxTree(repo,commits):
    logging.info("Enriching Commits with Syntax Tree Information")
    repoName = repo['full_name'].split('-')[1]
    repoNameFull = repoName.replace('-','/')
    if not path.exists("./"+repoName):
        repo_url = 'https://github.com/'+repoNameFull+'.git'
        repo_path = './'+repoName
        repo = clone_repository(repo_url, repo_path) # Clones a non-bare repository
    repo = pygit2.Repository('./'+repoName)
    for idx,commit in enumerate(commits):
        logging.info("Enriching Commit %d", idx)
        localCommit = repo.revparse_single(commit['sha'])
        patches = PatchSet.from_string(commit['diff'])
        commit['commonTypes'] = []
        for patch in patches:
            sourceCode = localCommit.tree[patch.path].read_raw()
            if(not patch.path.endswith('.java')):
                continue
            #print(patch.path)
            splittedSource = sourceCode.decode("utf-8").split('\n')
            tree = generateSyntaxtTree(sourceCode)
            for hunk in patch:
                # handle mentioning of hunk header (method name) in commit message
                for word in re.findall('[A-Za-z]+',hunk.section_header ):
                    if word in commit['common']:
                        for idx,line in enumerate(splittedSource):
                            if idx > hunk.target_start:
                                break
                            wordPosition = line.find(word)
                            if wordPosition != -1:
                                nodeType = findNodeTypeAtPosition(tree,idx,wordPosition,wordPosition+len(word))
                                commit['commonTypes'].append([nodeType,"header"])
                #handle program elements hunk content
                for line in hunk:
                    for word in re.findall('[A-Za-z]+',line.value):
                        if word in commit['common']:
                            wordPosition = line.value.find(word)
                            #TODO: build CST before the change
                            if line.line_type == '+':
                                nodeType = findNodeTypeAtPosition(tree,line.target_line_no-1,wordPosition,wordPosition+len(word))
                                lineType = ''
                                if line.line_type == ' ':
                                    lineType = "context"
                                elif line.line_type == '+':
                                    lineType = "added"
                                elif line.line_type == '-':
                                    lineType = "deleted"
                                commit['commonTypes'].append([nodeType,lineType])
                            


#get diff
#diff('HEAD', 'HEAD^')
#TODO Build a custom CST or mark some nodes as added or deleted. Maybe this is possible by altering the code of the tree sitter, by comparing the trees

#TODO Filter out words that are in comments

repoName = "ReactiveX/RxJava"
testRepo = {'full_name': repoName.replace('/','-'), 'contributors_url' :'https://api.github.com/repos/'+repoName+'/contributors', 'commits_url': 'https://api.github.com/repos/'+repoName+'/commits{/sha}', 'default_branch': '3.x', 'issues_url': 'https://api.github.com/repos/'+repoName+'/issues{/number}'}
commits = [{'sha': '17e71ab4e6dc432ac536bf5bf193055b32b3fb17', 'message': 'Updated assertValue and assertErrorMessage'}]
#commits = [{'sha': '17e71ab4e6dc432ac536bf5bf193055b32b3fb17', 'message': 'Updated the error message in assertValue and assertErrorMessage to show the exptected value'}]
diff = urllib.request.urlopen('https://github.com/ReactiveX/RxJava/commit/17e71ab4e6dc432ac536bf5bf193055b32b3fb17.diff')
commits[0]['diff'] = diff.read()
commits[0]['diff'] = commits[0]['diff'].decode('utf-8')

filtered_commits = analyzeRepo(testRepo,2,100)
commits = findCommitsThatContainProgramElements(testRepo, filtered_commits)
enrichCommitWithConcreteSyntaxTree(testRepo,commits)

for commit in commits:
    commit['diff'] = ''

writeCommitsToFile(testRepo['full_name']+".txt",commits)