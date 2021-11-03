from fastapi import FastAPI
import pygit2
from typing import List
from pydantic import BaseModel
import copy

orderedPatches = []
openPatches = []
filesToCommit = []
allFiles = []
app = FastAPI()

class Commit(BaseModel):
    message: str

class Files(BaseModel):
    filesList: List[str]

#order patches and hunks by most changes
def orderPatches(diff):
    global orderedPatches
    global openPatches
    global filesToCommit
    global allFiles
    orderedPatches = []
    filesToCommit = []
    allFiles = []
    for idx, patch in enumerate(diff):
        orderedPatches.append([idx,patch,[],patch.delta.new_file.path])
    orderedPatches = sorted(orderedPatches, key=lambda tuple: tuple[1].line_stats[0]+tuple[1].line_stats[1]+tuple[1].line_stats[2],  reverse=True)

    newOrderedPatches = []
    for (oldId,patch,hunks,path) in orderedPatches:
        allFiles.append(path)
        for idx, hunk in enumerate(patch.hunks):
            hunks.append((idx, len(hunk.lines)))
        newOrderedPatches.append([oldId,hunks,path])
    orderedPatches = newOrderedPatches

    for patch in orderedPatches:
        patch[1] = sorted(patch[1], key=lambda hunk: hunk[1],  reverse=True)

    openPatches = copy.deepcopy(orderedPatches)
    filesToCommit = copy.deepcopy(allFiles)
    print(orderedPatches)
    print("Biggest File",orderedPatches[0][2])
    print("Biggest File Index",orderedPatches[0][0])
    print("Biggest Hunk Index",orderedPatches[0][1][0][0])

#frontend remove the hunks to that files ==> alarm the user that questions that he already answered will be removed for that file

#updates the openHunks to reflect which files are selected by the user in the frontend 
@app.put("/filesToCommit")
async def filesToCom(filesSelectedByUser: Files):
    global filesToCommit
    global openPatches

    print("files from frontend ",filesSelectedByUser.filesList)
    print("files at backend ",filesToCommit)
    files = filesSelectedByUser.filesList
    addedFiles = []
    deletedFiles = []

    for path in files:
        if path in filesToCommit:
            filesToCommit.remove(path)
        else:
            addedFiles.append(path)
    
    deletedFiles = filesToCommit

    print("added ",addedFiles)
    print("delted ",deletedFiles)

    openPatches = [(oldId,hunks,path) for (oldId,hunks,path) in openPatches if path not in deletedFiles]
    print("open patches without deleted ",openPatches)
    for idx, (oldId,hunks,path) in enumerate(orderedPatches):
        if path in addedFiles:
            openPatches.append((oldId,hunks,path))
    
    openPatches = sorted(openPatches, key=lambda patch: patch[1],  reverse=True)
    print("new open patches", openPatches)
    filesToCommit = files
    print(filesToCommit)

@app.get("/getDiff")
async def getDiff():
    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")
    diff = repo.diff('HEAD', cached=False)

    orderPatches(diff)
    return diff.patch

@app.post("/commit")
async def commitFiles(commit: Commit):
    global filesToCommit
    print(filesToCommit)
    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")

    #clean index
    allFiles = ["README.md"]
    for path in allFiles:
        repo.index.remove(path)
        # Restore object from db
        obj = repo.revparse_single('HEAD').tree[path] # Get object from db
        repo.index.add(pygit2.IndexEntry(path, obj.oid, obj.filemode)) # Add to index

        # Write index
        repo.index.write()

    for path in filesToCommit:
        repo.index.add(path)
    repo.index.write()
    tree = repo.index.write_tree()
    parent, ref = repo.resolve_refish(refish=repo.head.name)
    repo.create_commit(
        ref.name,
        repo.default_signature,
        repo.default_signature,
        commit.message,
        tree,
        [parent.oid],
    )

@app.get("/getQuestions")
async def getQuestions(type:str = None, issues: List[int] = None, nextFile: bool = False):
    global openPatches

    needWhyQuestions = True
    if issues:
        needWhyQuestions = False

    print(nextFile)
    print(openPatches)
    if nextFile or len(orderedPatches[0][1]) == 0:
        del openPatches[0]

    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")

    # check if this is needed to also add untracked files
    repo.index.add_all([r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code"])
    repo.index.write()
    repo.index.write_tree()

    # How many Files changed in the Diff
    diff = repo.diff('HEAD', cached=False)
    if(diff.stats.files_changed == 0 or len(openPatches) == 0 or len(openPatches[0][1]) == 0):
        return {"question" :"Finsih"}
    # number of patches is normaly the same as number of files, i think there is a difference when the files do not contain changes that are diffable
    # then there is maybe no patch

    # a patch contains hunks, these hunks are the areas in the file that have changes

    for diffPatch in diff:
        print(diffPatch.line_stats)
        print("Patch has "+ str(len(diffPatch.hunks)) + " hunks")

    print(openPatches)

    nextHunk = {"question" :"This code part will",
            "file" : openPatches[0][0],
            "hunk" : openPatches[0][1][0][0],
            }

    del openPatches[0][1][0]
    
    # TODO: pre-process the diff by extracting program symbols
    # TODO: call the hunk ranker model

    return nextHunk