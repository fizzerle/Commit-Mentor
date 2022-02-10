from email import message
from fastapi import FastAPI, HTTPException
import pygit2
from typing import List,Dict,Tuple
from pydantic import BaseModel
import copy
from unidiff import PatchSet
import pkg_resources
import os
import subprocess
import time


orderedPatches = []
openPatches = []
filesToCommit = []
files = []
allFiles = []
patchesDiff = []
app = FastAPI()

diff = None
repo = None


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


# order patches and hunks by most changes


def orderPatches(diff):
    global orderedPatches
    global openPatches
    global filesToCommit
    global allFiles
    global patchesDiff
    orderedPatches = []
    filesToCommit = []
    allFiles = []
    for idx, patch in enumerate(diff):
        orderedPatches.append([idx, patch, [], patch.delta.new_file.path])
        patchesDiff.append(patch.text)
    orderedPatches = sorted(
        orderedPatches, key=lambda tuple: tuple[1].line_stats[0]+tuple[1].line_stats[1]+tuple[1].line_stats[2],  reverse=True)

    newOrderedPatches = []
    for (oldId, patch, hunks, path) in orderedPatches:
        allFiles.append(path)
        for idx, hunk in enumerate(patch.hunks):
            hunks.append((idx, len(hunk.lines)))
        newOrderedPatches.append([oldId, hunks, path])
    orderedPatches = newOrderedPatches

    for patch in orderedPatches:
        patch[1] = sorted(patch[1], key=lambda hunk: hunk[1],  reverse=True)

    openPatches = copy.deepcopy(orderedPatches)
    filesToCommit = copy.deepcopy(allFiles)
    print("ordered patches : ",orderedPatches)

# frontend remove the hunks to that files ==> alarm the user that questions that he already answered will be removed for that file

# updates the openHunks to reflect which files are selected by the user in the frontend


@app.put("/filesToCommit")
async def filesToCom(filesSelectedByUser: Files):
    global filesToCommit
    global openPatches

    print("files from frontend ", filesSelectedByUser.filesList)
    print("files at backend ", filesToCommit)
    files = filesSelectedByUser.filesList
    addedFiles = []
    deletedFiles = []

    for path in files:
        if path in filesToCommit:
            filesToCommit.remove(path)
        else:
            addedFiles.append(path)

    deletedFiles = filesToCommit

    print("added ", addedFiles)
    print("delted ", deletedFiles)

    openPatches = [(oldId, hunks, path) for (oldId, hunks, path)
                   in openPatches if path not in deletedFiles]
    print("open patches without deleted ", openPatches)
    for idx, (oldId, hunks, path) in enumerate(orderedPatches):
        if path in addedFiles:
            openPatches.append((oldId, hunks, path))

    openPatches = sorted(
        openPatches, key=lambda patch: patch[1],  reverse=True)
    print("new open patches", openPatches)
    filesToCommit = files
    print(filesToCommit)

@app.get("/getDiff")
async def getDiff():
    global diff
    global files
    global repo
    global patchesDiff

    repo = pygit2.Repository(
        r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")
    files = []

    status = repo.status()
    for entry in status:
        statusMode = ""
        if(status[entry] == pygit2.GIT_STATUS_WT_NEW or status[entry] == 1):
            statusMode = "NEW"
        if(status[entry] == pygit2.GIT_STATUS_WT_MODIFIED or status[entry] == 258 or status[entry] == 2):
            statusMode = "MODIFIED"
        if(status[entry] == pygit2.GIT_STATUS_WT_DELETED or status[entry] == 4):
            statusMode = "DELETED"
        if statusMode != "":
            files.append((entry,statusMode))
        print(entry,status[entry])

    unstageAllFiles()
    
    diff = repo.diff('HEAD', cached=False,flags =pygit2.GIT_DIFF_RECURSE_UNTRACKED_DIRS+pygit2.GIT_DIFF_INCLUDE_UNTRACKED+pygit2.GIT_DIFF_SHOW_UNTRACKED_CONTENT+pygit2.GIT_DIFF_SHOW_BINARY)
    #orderPatches(diff)
    #print(patchesDiff)

    return {'files':files,'diff':diff.patch}

def partialCommit(commitToPublish,uniDiffPatches):
    global repo
    os.chdir(r'C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code')
    for patch in commitToPublish.patches:
        diffToApply = ""
        uniDiffPatch = None
        #searching is needed because unidiff parsing changes the order
        for unidiffPat in uniDiffPatches:
            if patch.filename == unidiffPat.target_file[2:]:
                uniDiffPatch = unidiffPat
        fileStatus = repo.status()[patch.filename]
        if fileStatus == pygit2.GIT_STATUS_WT_NEW or fileStatus == pygit2.GIT_STATUS_WT_DELETED:
            continue
        filesToCommit.append(patch.filename)
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
            if hunk.answer == "":
                print(("hunk answer empty"))
                continue
            hunkPatch = diffToApply + str(uniDiffPatch[hunk.hunkNumber])
            print("hunk", hunk)
            print("hunkPatch generated:", hunkPatch)
            hunkPatch = hunkPatch.replace("\r\n", "\n").replace("\r", "\n")
            #https://stackoverflow.com/questions/10785131/line-endings-in-python
            with open("partial.patch", "w+", newline="") as text_file:
                text_file.write(hunkPatch)

            with open("partial.log", "w+") as text_file:
                process = subprocess.Popen(['git', 'apply', '--cached', '-v', r'C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code\partial.patch'],
                                    stdout=text_file, 
                                    stderr=text_file)
                process.communicate()

            #TODO: this would the more elegant solution to directly apply the diff in pygit2, however somhow this does not work
            #newDiff = pygit2.Diff.parse_diff(diffToApply)
            #repo.apply(newDiff,pygit2.GIT_APPLY_LOCATION_INDEX)

            repo.index.read()
            tree = repo.index.write_tree()
            parent, ref = repo.resolve_refish(refish=repo.head.name)
            repo.create_commit(
                ref.name,
                repo.default_signature,
                repo.default_signature,
                hunk.answer,
                tree,
                [parent.oid],
            )

def getFilesToAddAndToRemove(commitToPublish,patches):
    global repo
    wholeFilesToAdd = []
    wholeFilesToRemove = []
    for patch in commitToPublish.patches:
        diffToApply = ""
        uniDiffPatch = None
        #searching is needed because unidiff parsing changes the order
        for unidiffPat in patches:
            if patch.filename == unidiffPat.target_file[2:]:
                uniDiffPatch = unidiffPat
        fileStatus = repo.status()[patch.filename]
        if fileStatus == pygit2.GIT_STATUS_WT_NEW or fileStatus == pygit2.GIT_STATUS_WT_DELETED:
            if fileStatus == pygit2.GIT_STATUS_WT_NEW:
                wholeFilesToAdd.append(patch.filename)
            if fileStatus ==  pygit2.GIT_STATUS_WT_DELETED:
                wholeFilesToRemove.append(patch.filename)
            continue
    return (wholeFilesToAdd,wholeFilesToRemove)
def unstageAllFiles():
    global repo
    global files
    # unstage all files
    for (path,mode) in files:
        if(mode == "DELETED"):
            obj = repo.revparse_single('HEAD').tree[path]  # Get object from db
            repo.index.add(pygit2.IndexEntry(
            path, obj.oid, obj.filemode))
        if(path in repo.index):
            repo.index.remove(path)
            # Restore object from db
            if(path in repo.revparse_single('HEAD').tree):
                obj = repo.revparse_single('HEAD').tree[path]  # Get object from db
                repo.index.add(pygit2.IndexEntry(
                path, obj.oid, obj.filemode))  # Add to index
    repo.index.write()

@app.post("/commit")
async def commitFiles(commitToPublish: CommitToPublish):
    global filesToCommit
    global diff
    global files
    global repo

    diff = repo.diff('HEAD', cached=False)
    patches = PatchSet.from_string(diff.patch)
    status = repo.status()

    print(commitToPublish)


    partialCommit(commitToPublish,patches)
    wholeFilesToAdd,wholeFilesToRemove = getFilesToAddAndToRemove(commitToPublish,patches)

    for patch in commitToPublish.patches:
        if patch.filename in wholeFilesToAdd:
            repo.index.add(patch.filename)
        elif patch.filename in wholeFilesToRemove:
            repo.index.remove(patch.filename)
        else:
            continue
    repo.index.write()
    tree = repo.index.write_tree()
    parent, ref = repo.resolve_refish(refish=repo.head.name)
    repo.create_commit(
        ref.name,
        repo.default_signature,
        repo.default_signature,
        commitToPublish.message,
        tree,
        [parent.oid],
    )


@app.get("/getQuestions")
async def getQuestions(type: str = None, issues: List[int] = None, nextFile: bool = False):
    global openPatches
    global diff

    needWhyQuestions = True
    if issues:
        needWhyQuestions = False
    # How many Files changed in the Diff
    if(diff.stats.files_changed == 0 or len(openPatches) == 0):
        return {"question": "Finsih"}
    print("openPatches before delete", openPatches)
    if nextFile or len(openPatches[0][1]) == 0:
            del openPatches[0]
    print("openPatches after delete", openPatches)
    if(diff.stats.files_changed == 0 or len(openPatches) == 0):
        return {"question": "Finsih"}


    for diffPatch in diff:
        print(diffPatch.line_stats)
        print("Patch has " + str(len(diffPatch.hunks)) + " hunks")

    print("get Questio OPEN Patches",openPatches)
    print("get Questio ORDERED Patches",orderedPatches)

    hunkCount = 0
    for patch in orderedPatches:
        if patch[0] == openPatches[0][0]:
            hunkCount = len(patch[1])

    nextHunk = {"question": "This code part will",
                "fileNumber": openPatches[0][0],
                "filePath": openPatches[0][2],
                "hunkNumber": openPatches[0][1][0][0],
                "openFiles": len(openPatches),
                "openHunks":len(openPatches[0][1]),
                "allHunksForThisFile": hunkCount
                }

    del openPatches[0][1][0]

    # TODO: pre-process the diff by extracting program symbols
    # TODO: call the hunk ranker model

    return nextHunk
