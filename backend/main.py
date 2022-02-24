from fileinput import filename
from fastapi import FastAPI, HTTPException
import pygit2
from typing import List
from pydantic import BaseModel
import copy
from unidiff import PatchSet
import os
import subprocess
import re
from allennlp.predictors import Predictor
import logging

orderedPatches = []
openPatches = []
filesToCommit = []
files = []
allFiles = []
patchesDiff = []
app = FastAPI()

diff = None
repo = None
diffClean = None

projectPath = ""


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
def orderPatches(diff):
    logging.info("Enter orderPatches method")
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
        if(len(patch.hunks)) == 0:
            hunks.append((0, 0))
        newOrderedPatches.append([oldId, hunks, path])
    orderedPatches = newOrderedPatches

    for patch in orderedPatches:
        patch[1] = sorted(patch[1], key=lambda hunk: hunk[1],  reverse=True)

    openPatches = copy.deepcopy(orderedPatches)
    filesToCommit = copy.deepcopy(allFiles)
    logging.info("Ordered patches: %s", orderedPatches)

# frontend remove the hunks to that files ==> alarm the user that questions that he already answered will be removed for that file
# updates the openHunks to reflect which files are selected by the user in the frontend
@app.put("/filesToCommit")
async def filesToCom(filesSelectedByUser: Files):
    logging.info('The user selected files at frontend')
    global filesToCommit
    global openPatches

    logging.info("Files selected at frontend %s", filesSelectedByUser.filesList)
    logging.info("Last know selected Files at backend %s", filesToCommit)
    files = filesSelectedByUser.filesList
    addedFiles = []
    deletedFiles = []

    for path in files:
        if path in filesToCommit:
            filesToCommit.remove(path)
        else:
            addedFiles.append(path)

    deletedFiles = filesToCommit

    logging.info("These Files where added %s", addedFiles)
    logging.info("These Files where deleted %s", deletedFiles)

    openPatches = [(oldId, hunks, path) for (oldId, hunks, path)
                   in openPatches if path not in deletedFiles]
    logging.info("Open Patches without deleted %s", openPatches)
    for idx, (oldId, hunks, path) in enumerate(orderedPatches):
        if path in addedFiles:
            openPatches.append((oldId, hunks, path))

    openPatches = sorted(
        openPatches, key=lambda patch: patch[1],  reverse=True)
    logging.info("Open Patches with newly added files %s", openPatches)
    filesToCommit = files
    print(filesToCommit)

@app.get("/getDiff")
async def getDiff(path: str):
    global diff
    global files
    global repo
    global patchesDiff
    global diffClean
    global projectPath
    if(os.path.isdir(path)):
        projectPath = path
        try:
            repo = pygit2.Repository(projectPath)
        except pygit2.GitError:
            raise HTTPException(status_code=404, detail="The directory does not contain a git repository")
    else:
        raise HTTPException(status_code=404, detail="Directory not found")
    repo = pygit2.Repository(projectPath)
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
        logging.debug(entry,status[entry])

    unstageAllFiles()
    diff = repo.diff('HEAD', cached=False,flags =pygit2.GIT_DIFF_RECURSE_UNTRACKED_DIRS+pygit2.GIT_DIFF_INCLUDE_UNTRACKED+pygit2.GIT_DIFF_SHOW_UNTRACKED_CONTENT)
    logging.info("Got diff with untracked files")
    diffClean = repo.diff('HEAD', cached=False)
    logging.info("Got diff without untracked files")
    orderPatches(diff)

    return {'files':files,'diff':diff.patch}

def unstageAllFiles():
    logging.info("Unstaged all files")
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

'''
all this code is need because the pygit2 patches do not contain enough information to generate a valid partial patch file
so i need the unidiff patch library to collect all the information and i have to map between the unidiff patches and the
pygit2 patches which have different order after parsing
'''
def partialCommit(commitToPublish,uniDiffPatches):
    logging.info("Enter partial commit")
    global repo
    global projectPath
    os.chdir(projectPath)
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
                process = subprocess.Popen(['git', 'apply', '--cached', '-v', projectPath+"\partial.patch"],
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
def getFilesToAddAndToRemove(commitToPublish,patches):
    global repo
    wholeFilesToAdd = []
    wholeFilesToRemove = []
    for patch in commitToPublish.patches:
        fileStatus = repo.status()[patch.filename]
        if fileStatus == pygit2.GIT_STATUS_WT_NEW or fileStatus == pygit2.GIT_STATUS_WT_DELETED:
            if fileStatus == pygit2.GIT_STATUS_WT_NEW:
                wholeFilesToAdd.append(patch.filename)
            if fileStatus ==  pygit2.GIT_STATUS_WT_DELETED:
                wholeFilesToRemove.append(patch.filename)
            continue
    return (wholeFilesToAdd,wholeFilesToRemove)

@app.post("/commit")
async def commitFiles(commitToPublish: CommitToPublish):
    global filesToCommit
    global diff
    global files
    global repo
    global diffClean
    logging.info("Clean diff: %s",diffClean.patch)
    patches = PatchSet.from_string(diffClean.patch)

    logging.info("Commit to publish: %s",commitToPublish)


    partialCommit(commitToPublish,patches)
    wholeFilesToAdd,wholeFilesToRemove = getFilesToAddAndToRemove(commitToPublish,patches)

    #commit files that are either new or deleted
    repo.index.read()
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
async def getQuestions(nextFile: bool = False):
    global openPatches
    global diff


    logging.info("Open Patches are: %s",openPatches)
    # when there are no hunks left
    if(diff.stats.files_changed == 0 or len(openPatches) == 0):
        logging.info("send Finish because no questions left")
        return {"question": "Finsih"}
    if nextFile or len(openPatches[0][1]) == 0:
        del openPatches[0]
    logging.info("Open Patches after delete: %s", openPatches)

    # when somebody skipped the file and no hunks are left
    if(diff.stats.files_changed == 0 or len(openPatches) == 0):
        logging.info("send Finish because no questions left")
        return {"question": "Finsih"}

    logging.info("Ordered Patches: %s",orderedPatches)

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

    return nextHunk

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)


class ModelConfig:
    batch_size = 32
    output_size = 2
    hidden_dim = 384
    n_layers = 2
    lr = 2e-5
    bidirectional = True
    drop_prob = 0.55
    # training params
    epochs = 10
    print_every = 10
    clip = 5  # gradient clipping
    use_cuda = USE_CUDA
    bert_path = 'bert-base-uncased'
    save_path = './bert_bilstm.pth'
    sampleRate = 2
    labelSelected = 2  # 2/3, 2:Why label; 3:What label


class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.bert = BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = self.bert(x)[0]
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)

        if self.bidirectional:
            hidden_last_L = hidden_last[-2]
            hidden_last_R = hidden_last[-1]
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        else:
            hidden_last_out = hidden_last[-1]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )
        return hidden



def test_model(input):
    global h
    global net
    output = net(input, h)
    logging.info("Model predicted: %s", output)
    # output = torch.nn.Softmax(dim=1)(output)
    _, predMax = torch.max(output, 1)
    logging.info("Response Score: %s",output.data[0][1].data.item())
    return output.data[0][1].data.item()

tokenizer = None
h = None
net = None

@app.post("/checkMessage")
async def checkMessage(commitToPublish: CommitToPublish):
    global tokenizer
    global diffClean
    #message = "Issue <issue_link> ; when arrays differ in length, say so, but go ahead and find the first difference as usual to ease diagnosis"
    message = commitToPublish.message
    logging.info("Original Commit message: %s",message)

    patches = PatchSet.from_string(diffClean.patch)

    uniDiffPatches = []
    filePaths = []
    for patch in commitToPublish.patches:

        #searching is needed because unidiff parsing changes the order
        for unidiffPat in patches:
            if patch.filename == unidiffPat.target_file[2:]:
                uniDiffPatches.append(unidiffPat)
        filePaths.append(patch.filename)
    message = preprocessMessageForModel(message,patches,filePaths)

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
