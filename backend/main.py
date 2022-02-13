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
        print(entry,status[entry])

    unstageAllFiles()
    
    diff = repo.diff('HEAD', cached=False,flags =pygit2.GIT_DIFF_RECURSE_UNTRACKED_DIRS+pygit2.GIT_DIFF_INCLUDE_UNTRACKED+pygit2.GIT_DIFF_SHOW_UNTRACKED_CONTENT+pygit2.GIT_DIFF_SHOW_BINARY)
    diffClean = repo.diff('HEAD', cached=False)
    orderPatches(diff)

    return {'files':files,'diff':diff.patch}

def partialCommit(commitToPublish,uniDiffPatches):
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
            print("hunkNumber", hunk.hunkNumber)
            hunkPatch = diffToApply + str(uniDiffPatch[hunk.hunkNumber])
            print("hunk", hunk)
            print("hunkPatch generated:", hunkPatch)
            hunkPatch = hunkPatch.replace("\r\n", "\n").replace("\r", "\n")
            #https://stackoverflow.com/questions/10785131/line-endings-in-python
            with open("partial.patch", "w+", newline="") as text_file:
                text_file.write(hunkPatch)

            with open("partial.log", "w+") as text_file:
                process = subprocess.Popen(['git', 'apply', '--cached', '-v', projectPath+"\partial.patch"],
                                    stdout=text_file, 
                                    stderr=text_file)
                process.communicate()

            #TODO: this would the more elegant solution to directly apply the diff in pygit2, however somhow this does not work
            #newDiff = pygit2.Diff.parse_diff(diffToApply)
            #repo.apply(newDiff,pygit2.GIT_APPLY_LOCATION_INDEX)

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
    global diffClean

    patches = PatchSet.from_string(diffClean.patch)

    print(commitToPublish)


    partialCommit(commitToPublish,patches)
    wholeFilesToAdd,wholeFilesToRemove = getFilesToAddAndToRemove(commitToPublish,patches)

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
async def getQuestions(type: str = None, issues: List[int] = None, nextFile: bool = False):
    global openPatches
    global diff

    needWhyQuestions = True
    if issues:
        needWhyQuestions = False
    print(diff.stats.files_changed)
    print(openPatches)
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

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
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
    print("Model predicted: ", output)
    # output = torch.nn.Softmax(dim=1)(output)
    _, predMax = torch.max(output, 1)
    min, max = torch.aminmax(output)
    diff = max - min
    print("Model predicted: ", predMax.cpu().numpy()[0])
    print("Difference of two values is: ", diff.cpu().detach().numpy())

    if(diff <= 2):
      print("I am unsure")
    if(predMax.cpu().numpy()[0] == 1):
      print("Contains Why")
    if(predMax.cpu().numpy()[0] == 0):
      print("Contains NO Why")

tokenizer = None
h = None
net = None

@app.get("/questionQuality")
async def getQuestions(message: str = None):
    global tokenizer
    #message = "Issue <issue_link> ; when arrays differ in length, say so, but go ahead and find the first difference as usual to ease diagnosis"
    print(message)
    message.replace('<enter>', '$enter').replace('<tab>', '$tab'). \
                                    replace('<url>', '$url').replace('<version>', '$version') \
                                    .replace('<pr_link>', '$pull request>').replace('<issue_link >',
                                                                                    '$issue') \
                                    .replace('<otherCommit_link>', '$other commit').replace("<method_name>",
                                                                                            "$method") \
                                    .replace("<file_name>", "$file").replace("<iden>", "$token")
    


    message_tokens = tokenizer(message,
                                padding=True,
                                truncation=True,
                                max_length=200,
                                return_tensors='pt')
    X = message_tokens['input_ids']

    if (USE_CUDA):
        print('Run on GPU.')
    else:
        print('No GPU available, run on CPU.')

    test_model(X)


@app.on_event("startup")
async def setupTokenizerAndModel():
    global model_config
    global h
    global net
    global tokenizer
    model_config = ModelConfig()
    tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)
    if model_config.labelSelected == 2:
        print("Why Message")
    else:
        print("What Message")

    net = bert_lstm(model_config.bert_path,
                    model_config.hidden_dim,
                    model_config.output_size,
                    model_config.n_layers,
                    model_config.bidirectional,
                    model_config.drop_prob
                    )
    net.load_state_dict(torch.load(model_config.save_path,map_location=torch.device('cpu')))
    if (model_config.use_cuda):
        net.cuda()
    net.eval()

    # init hidden state
    h = net.init_hidden(1)

    net.eval()

    h = tuple([each.data for each in h])