from fileinput import filename
from fastapi import FastAPI, HTTPException
import pygit2
from typing import List
from pydantic import BaseModel
import copy
from unidiff import PatchSet
import os
import subprocess
from allennlp.predictors import Predictor
import logging
from preprocessMessage import preprocessMessageForModel
from messageQualityModel import test_model, ModelConfig, setupModel

app = FastAPI()

commitProcess = None

class CommitProcess:
    orderedPatches = []
    openPatches = []
    filesSelectedByTheUser = []
    fileNamesWithNewAndDeleted = []
    projectPath = ""

    pyGit2Diff = None
    pyGit2Repository = None

class Commit(BaseModel):
    message: str

class Files(BaseModel):
    filesList: List[str]

class Hunk(BaseModel):
    hunkNumber: int
    answer: str
class Patch(BaseModel):
    filename: str
    patchNumber: int
    hunks: List[Hunk]
class CommitToPublish(BaseModel):
    message: str
    patches: List[Patch]

class QuestionAnswerPair(BaseModel):
    question: str
    answer: str
class DiaryQuestions(BaseModel):
    date: str
    numberOfCommits: int
    secondsSpentCommiting: int
    secondsSpentAnsweringHunks: int
    secondsSpentAddingTheRationale: int
    commitMessageLength: str
    diffLength: str
    issuesLinked: bool
    answers: List[QuestionAnswerPair]


# order patches and hunks by most changes
def orderPatches(commitProcess):
    logging.info("Enter orderPatches method")
    orderedPatches = []
    allFiles = []
    for idx, patch in enumerate(commitProcess.pyGit2Diff):
        orderedPatches.append([idx, patch, [], patch.delta.new_file.path])
    orderedPatches = sorted(
        orderedPatches, key=lambda tuple: tuple[1].line_stats[0]+tuple[1].line_stats[1]+tuple[1].line_stats[2],  reverse=True)

    newOrderedPatches = []
    for (oldId, patch, hunks, path) in orderedPatches:
        allFiles.append(path)
        for idx, hunk in enumerate(patch.hunks):
            hunks.append((idx, len(hunk.lines)))
        if(len(patch.hunks)) == 0:
            hunks.append((0, 0))
        newOrderedPatches.append([oldId, hunks, path])
    commitProcess.orderedPatches = newOrderedPatches

    for patch in newOrderedPatches:
        patch[1] = sorted(patch[1], key=lambda hunk: hunk[1],  reverse=True)

    commitProcess.openPatches = copy.deepcopy(newOrderedPatches)
    commitProcess.filesSelectedByTheUser = allFiles
    logging.info("Ordered patches: %s", orderedPatches)

# frontend remove the hunks to that files ==> alarm the user that questions that he already answered will be removed for that file
# updates the openHunks to reflect which files are selected by the user in the frontend
@app.put("/filesToCommit")
async def filesToCom(filesSelectedByUser: Files):
    logging.info('The user selected files at frontend')
    global commitProcess

    logging.info("Files selected at frontend %s", filesSelectedByUser.filesList)
    logging.info("Last know selected Files at backend %s", commitProcess.filesSelectedByTheUser)
    files = filesSelectedByUser.filesList
    addedFiles = []
    deletedFiles = []

    for path in files:
        if path in commitProcess.filesSelectedByTheUser:
            commitProcess.filesSelectedByTheUser.remove(path)
        else:
            addedFiles.append(path)

    deletedFiles = commitProcess.filesSelectedByTheUser

    logging.info("These Files where added %s", addedFiles)
    logging.info("These Files where deleted %s", deletedFiles)

    commitProcess.openPatches = [(oldId, hunks, path) for (oldId, hunks, path)
                   in commitProcess.openPatches if path not in deletedFiles]
    logging.info("Open Patches without deleted %s", commitProcess.openPatches)
    for idx, (oldId, hunks, path) in enumerate(commitProcess.orderedPatches):
        if path in addedFiles:
            commitProcess.openPatches.append((oldId, hunks, path))

    commitProcess.openPatches = sorted(
        commitProcess.openPatches, key=lambda patch: patch[1],  reverse=True)
    logging.info("Open Patches with newly added files %s", commitProcess.openPatches)
    logging.info("Files that are selected %s", commitProcess.filesSelectedByTheUser)
    commitProcess.filesSelectedByTheUser = files
    return len(commitProcess.openPatches)

@app.get("/getDiff")
async def getDiff(path: str):
    global commitProcess

    commitProcess = CommitProcess()
    if(os.path.isdir(path)):
        try:
            commitProcess.pyGit2Repository = pygit2.Repository(path)
        except pygit2.GitError:
            raise HTTPException(status_code=404, detail="The directory does not contain a git repository")
    else:
        raise HTTPException(status_code=404, detail="Directory not found")
    
    commitProcess.projectPath = path
    commitProcess.pyGit2Repository = pygit2.Repository(path)
    commitProcess.fileNamesWithNewAndDeleted = []

    status = commitProcess.pyGit2Repository.status()
    for entry in status:
        statusMode = ""
        if(status[entry] == pygit2.GIT_STATUS_WT_NEW or status[entry] == 1):
            statusMode = "NEW"
        if(status[entry] == pygit2.GIT_STATUS_WT_MODIFIED or status[entry] == 258 or status[entry] == 2):
            statusMode = "MODIFIED"
        if(status[entry] == pygit2.GIT_STATUS_WT_DELETED or status[entry] == 4):
            statusMode = "DELETED"
        if statusMode != "":
            commitProcess.fileNamesWithNewAndDeleted.append((entry,statusMode))
        logging.debug(entry,status[entry])

    unstageAllFiles()
    commitProcess.pyGit2Diff = commitProcess.pyGit2Repository.diff('HEAD', cached=False,flags =pygit2.GIT_DIFF_RECURSE_UNTRACKED_DIRS+pygit2.GIT_DIFF_INCLUDE_UNTRACKED+pygit2.GIT_DIFF_SHOW_UNTRACKED_CONTENT)
    logging.info("Got diff with untracked files")
    orderPatches(commitProcess)

    return {'files':commitProcess.fileNamesWithNewAndDeleted,'diff':commitProcess.pyGit2Diff.patch}

def unstageAllFiles():
    logging.info("Unstaged all files")
    global commitProcess

    # unstage all files
    for (path,mode) in commitProcess.fileNamesWithNewAndDeleted:
        if(mode == "DELETED"):
            obj = commitProcess.pyGit2Repository.revparse_single('HEAD').tree[path]  # Get object from db
            commitProcess.pyGit2Repository.index.add(pygit2.IndexEntry(
            path, obj.oid, obj.filemode))
        if(path in commitProcess.pyGit2Repository.index):
            commitProcess.pyGit2Repository.index.remove(path)
            # Restore object from db
            if(path in commitProcess.pyGit2Repository.revparse_single('HEAD').tree):
                obj = commitProcess.pyGit2Repository.revparse_single('HEAD').tree[path]  # Get object from db
                commitProcess.pyGit2Repository.index.add(pygit2.IndexEntry(
                path, obj.oid, obj.filemode))  # Add to index
    commitProcess.pyGit2Repository.index.write()

'''
all this code is need because the pygit2 patches do not contain enough information to generate a valid partial patch file
so i need the unidiff patch library to collect all the information and i have to map between the unidiff patches and the
pygit2 patches which have different order after parsing
'''
def partialCommit(commitToPublish,uniDiffPatches):
    logging.info("Enter partial commit")
    global commitProcess
    filesSelectedByTheUser = commitProcess.filesSelectedByTheUser
    os.chdir(commitProcess.projectPath)
    for patch in commitToPublish.patches:
        diffToApply = ""
        uniDiffPatch = None
        #searching is needed because unidiff parsing changes the order
        for unidiffPat in uniDiffPatches:
            if patch.filename == unidiffPat.target_file[2:]:
                uniDiffPatch = unidiffPat
        fileStatus = commitProcess.pyGit2Repository.status()[patch.filename]
        if fileStatus == pygit2.GIT_STATUS_WT_NEW or fileStatus == pygit2.GIT_STATUS_WT_DELETED:
            continue
        filesSelectedByTheUser.append(patch.filename)
        source = ""
        target = ""
        if not uniDiffPatch.is_binary_file:
            source = "--- %s%s\n" % (
                uniDiffPatch.source_file,
                '\t' + uniDiffPatch.source_timestamp if uniDiffPatch.source_timestamp else '')
            target = "+++ %s%s\n" % (
                uniDiffPatch.target_file,
                '\t' + uniDiffPatch.target_timestamp if uniDiffPatch.target_timestamp else '')
        diffToApply += str(uniDiffPatch.patch_info) + source + target

        for hunk in patch.hunks:
            hunkPatch = diffToApply + str(uniDiffPatch[hunk.hunkNumber])
            logging.info("Hunk that gets applied %s", hunk)
            # This is need because: https://stackoverflow.com/questions/10785131/line-endings-in-python
            hunkPatch = hunkPatch.replace("\r\n", "\n").replace("\r", "\n")

            '''
            this is a workaround because pygit2 apply method has a bug
            so i have to write the diff to a file and then call git via a subprocess to apply the diff
            in the file partial log there can be found the error messages of git, if git could not apply the diff
            '''
            with open("partial.patch", "w+", encoding='utf-8', newline="") as text_file:
                text_file.write(hunkPatch)

            with open("partial.log", "w+",encoding='utf-8') as text_file:
                process = subprocess.Popen(['git', 'apply', '--cached', '-v', commitProcess.projectPath+"\partial.patch"],
                                    stdout=text_file, 
                                    stderr=text_file)
                process.communicate()
            
            '''
            TODO: this would the more elegant solution to directly apply the diff in pygit2, however somhow this does not work
            newDiff = pygit2.Diff.parse_diff(diffToApply)
            repo.apply(newDiff,pygit2.GIT_APPLY_LOCATION_INDEX)
            '''

'''
returns the filenames of the files that are new or got deleted
'''
#TODO: MAYBE THIS IS THE PROBLEM WITH NEW OR DELETED FILES, WRONG PATCHES USED
def getFilesToAddAndToRemove(commitToPublish,patches):
    global commitProcess
    wholeFilesToAdd = []
    wholeFilesToRemove = []
    for patch in commitToPublish.patches:
        fileStatus = commitProcess.pyGit2Repository.status()[patch.filename]
        if fileStatus == pygit2.GIT_STATUS_WT_NEW or fileStatus == pygit2.GIT_STATUS_WT_DELETED:
            if fileStatus == pygit2.GIT_STATUS_WT_NEW:
                wholeFilesToAdd.append(patch.filename)
            if fileStatus ==  pygit2.GIT_STATUS_WT_DELETED:
                wholeFilesToRemove.append(patch.filename)
            continue
    return (wholeFilesToAdd,wholeFilesToRemove)

@app.post("/commit")
async def commitFiles(commitToPublish: CommitToPublish):
    global commitProcess
    patches = PatchSet.from_string(commitProcess.pyGit2Diff.patch)

    logging.info("Commit to publish: %s",commitToPublish)


    partialCommit(commitToPublish,patches)
    wholeFilesToAdd,wholeFilesToRemove = getFilesToAddAndToRemove(commitToPublish,patches)

    #commit files that are either new or deleted
    commitProcess.pyGit2Repository.index.read()
    for patch in commitToPublish.patches:
        if patch.filename in wholeFilesToAdd:
            commitProcess.pyGit2Repository.index.add(patch.filename)
        elif patch.filename in wholeFilesToRemove:
            commitProcess.pyGit2Repository.index.remove(patch.filename)
        else:
            continue
    commitProcess.pyGit2Repository.index.write()
    tree = commitProcess.pyGit2Repository.index.write_tree()
    parent, ref = commitProcess.pyGit2Repository.resolve_refish(refish=commitProcess.pyGit2Repository.head.name)
    commitProcess.pyGit2Repository.create_commit(
        ref.name,
        commitProcess.pyGit2Repository.default_signature,
        commitProcess.pyGit2Repository.default_signature,
        commitToPublish.message,
        tree,
        [parent.oid],
    )


@app.get("/getQuestions")
async def getQuestions(nextFile: bool = False):
    global commitProcess

    logging.info("Open Patches are: %s",commitProcess.openPatches)
    # when there are no hunks left
    if(commitProcess.pyGit2Diff.stats.files_changed == 0 or len(commitProcess.openPatches) == 0):
        logging.info("send Finish because no questions left")
        return {"question": "Finsih"}
    if nextFile or len(commitProcess.openPatches[0][1]) == 0:
        del commitProcess.openPatches[0]
    logging.info("Open Patches after delete: %s", commitProcess.openPatches)

    # when somebody skipped the file and no hunks are left
    if(commitProcess.pyGit2Diff.stats.files_changed == 0 or len(commitProcess.openPatches) == 0):
        logging.info("send Finish because no questions left")
        return {"question": "Finsih"}

    logging.info("Ordered Patches: %s",commitProcess.orderedPatches)

    hunkCount = 0
    for patch in commitProcess.orderedPatches:
        if patch[0] == commitProcess.openPatches[0][0]:
            hunkCount = len(patch[1])

    nextHunk = {"question": "This code part will",
                "fileNumber": commitProcess.openPatches[0][0],
                "filePath": commitProcess.openPatches[0][2],
                "hunkNumber": commitProcess.openPatches[0][1][0][0],
                "openFiles": len(commitProcess.openPatches),
                "openHunks":len(commitProcess.openPatches[0][1]),
                "allHunksForThisFile": hunkCount
                }
    del commitProcess.openPatches[0][1][0]
    logging.info("Next hunk returned: %s",nextHunk)
    return nextHunk

@app.post("/checkMessage")
async def checkMessage(commitToPublish: CommitToPublish):
    global tokenizer
    global commitProcess
    global predictor
    global net
    global h
    #message = "Issue <issue_link> ; when arrays differ in length, say so, but go ahead and find the first difference as usual to ease diagnosis"
    message = commitToPublish.message
    logging.info("Original Commit message: %s",message)

    patches = PatchSet.from_string(commitProcess.pyGit2Diff.patch)

    filePaths = []
    for patch in commitToPublish.patches:
        filePaths.append(patch.filename)
    message = preprocessMessageForModel(message,patches,filePaths, predictor)

    logging.info("Preprocessed Commit message: %s",message)
    message_tokens = tokenizer(message,
                                padding=True,
                                truncation=True,
                                max_length=200,
                                return_tensors='pt')
    X = message_tokens['input_ids']

    return test_model(X, net, h)

tokenizer = None
h = None
net = None
predictor = None

@app.on_event("startup")
async def setupTokenizerAndModel():
    global tokenizer
    global predictor
    global net
    global h
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    # Predictor used for preprocessing steps
    predictor = Predictor.from_path("./tools/elmo-constituency-parser-2020.02.10.tar.gz")
    #other method to not load the predictor from local archive
    #predictor = pretrained.load_predictor("structured-prediction-constituency-parser")

    net, h, tokenizer = setupModel()
