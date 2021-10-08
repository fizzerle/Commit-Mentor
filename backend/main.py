from fastapi import FastAPI
import pygit2
from typing import List
from pydantic import BaseModel

app = FastAPI()

class Commit(BaseModel):
    message: str

@app.get("/getDiff")
async def getDiff():
    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")
    diff = repo.diff('HEAD', cached=False).patch
    return diff

@app.post("/commit")
async def commitFiles(commit: Commit):
    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")
    repo.index.add_all()
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
async def getQuestions(type:str = None, issues: List[int] = None):
    needWhyQuestions = True
    if issues:
        needWhyQuestions = False

    repo=pygit2.Repository(r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code")

    # check if this is needed to also add untracked files
    repo.index.add_all([r"C:\Users\Thomas\Dropbox\INFO Studium\Master\Masterarbeit\Code"])
    repo.index.write()
    repo.index.write_tree()

    # How many Files changed in the Diff
    diff = repo.diff('HEAD', cached=False)
    print(diff.stats.files_changed)
    # number of patches is normaly the same as number of files, i think there is a difference when the files do not contain changes that are diffable
    # then there is maybe no patch
    firstPatch = diff[0]
    # a patch contains hunks, these hunks are the areas in the file that have changes

    print(firstPatch.text)
    questions = []
    content = ""
    for diffPatch in diff:
        print(diffPatch.line_stats)
        print("Patch has "+ str(len(diffPatch.hunks)) + " hunks")
        for hunk in diffPatch.hunks:
            questions.append(hunk)
        #print(diffPatch.hunks[0].header)
        print(diffPatch.delta.new_file.path)
        print(diffPatch.delta.old_file.path)
        content += diffPatch.hunks[0].header+" "
        for line in diffPatch.hunks[0].lines:
            content += line.content + " "
    nextHunk = {"question" :"What is this hunk about?",
            "file" :  0,
            "hunk" : 0,
            }
    
    # TODO: pre-process the diff by extracting program symbols
    # TODO: call the hunk ranker model

    return nextHunk