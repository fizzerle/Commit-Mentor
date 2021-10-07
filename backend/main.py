# main.py

from fastapi import FastAPI
import pygit2
from typing import List

app = FastAPI()

@app.get("/getDiff")
async def getDiff():
    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")
    diff = repo.diff('HEAD', cached=True).patch
    return diff

@app.get("/getQuestions/")
async def getQuestions(type: type = None, issues: List[int] = None):
    needWhyQuestions = True
    if issues:
        needWhyQuestions = False

    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")

    # check if this is needed to also add untracked files
    repo.index.add_all([r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code"])
    repo.index.write()
    repo.index.write_tree()

    # How many Files changed in the Diff
    diff = repo.diff('HEAD', cached=True)
    print(diff.stats.files_changed)
    # number of patches is normaly the same as number of files, i think there is a difference when the files do not contain changes that are diffable
    # then there is maybe no patch
    firstPatch = diff[0]
    # a patch contains hunks, these hunks are the areas in the file that have changes

    print(firstPatch.text)
    for diffPatch in diff:
        print(diffPatch.line_stats)
        print("Patch has "+ str(len(diffPatch.hunks)) + " hunks")
        #print(diffPatch.hunks[0].header)
        #for line in diffPatch.hunks[0].lines:
        #    print(line.content)
    
    # TODO: pre-process the diff by extracting program symbols
    # TODO: call the hunk ranker model

    return "diff"