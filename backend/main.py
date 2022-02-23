from email import message
from fileinput import filename
from fastapi import FastAPI, HTTPException
import pygit2
from typing import List,Dict,Tuple
from pydantic import BaseModel
import copy
from unidiff import PatchSet
import pkg_resources
import os
import subprocess
import re
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive


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
            print("hunkNumber", hunk.hunkNumber)
            hunkPatch = diffToApply + str(uniDiffPatch[hunk.hunkNumber])
            print("hunk", hunk)
            print("hunkPatch generated:", hunkPatch)
            hunkPatch = hunkPatch.replace("\r\n", "\n").replace("\r", "\n")
            #https://stackoverflow.com/questions/10785131/line-endings-in-python
            with open("partial.patch", "w+", encoding='utf-8', newline="") as text_file:
                text_file.write(hunkPatch)

            with open("partial.log", "w+",encoding='utf-8') as text_file:
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

    if (USE_CUDA):
        logging.info('Run on GPU.')
    else:
        logging.info('No GPU available, run on CPU.')

    return test_model(X)




@app.on_event("startup")
async def setupTokenizerAndModel():
    global model_config
    global h
    global net
    global tokenizer
    global predictor
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    model_config = ModelConfig()
    predictor = Predictor.from_path("./tools/elmo-constituency-parser-2020.02.10.tar.gz")
    #predictor = pretrained.load_predictor("structured-prediction-constituency-parser")
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

from allennlp_models import pretrained

def preprocessMessageForModel(message,patches,filepaths):
    global predictor
    message = find_url(message)
    message = find_version(message)
    message = find_rawCode(message)
    message = find_SignInfo(message)
    message = find_IssueLinks(message)

    if message.strip(" ") == "":
        message = "empty log message"
    message = replace_file_name(message,filepaths)

    tokens, tags, length = allennlp_tag(message, predictor)

    indices, tokens = filter_tokens(length, tokens, tags)
    if len(indices) > 0:
        fount_indices, found_tokens = search_in_patches(patches, indices, tokens)
        if len(fount_indices) > 0:
            message = replace_tokens(message, found_tokens)

    message.replace('<enter>', '$enter').replace('<tab>', '$tab').\
    replace('<url>', '$url').replace('<version>', '$versionNumber')\
    .replace('<pr_link>','$pullRequestLink>').replace('<issue_link >','$issueLink')\
    .replace('<otherCommit_link>','$otherCommitLink').replace("<method_name>","$methodName")\
    .replace("<file_name>","$fileName").replace("<iden>","$token")

    return message

def cmp(elem):
    return elem[0]

def replace_file_name(message,filepaths):
    # replaced_tokens = find_file_name(sample)
    replaced_tokens = find_file_name2(message,filepaths)
    message = message

    # find out start and end index of replaced tokens
    locations = []
    diffMeanPunctuations = ['@']
    for t in replaced_tokens:
        end = 0
        while end<len(message):
            start = str(message).find(t, end, len(message))
            if start == -1:
                break
            end = start + len(t)
            before = start > 0 and (str(message[start-1]).isalnum() or str(message[start-1]) in diffMeanPunctuations)
            after = end < len(message) and str(message[end]).isalnum()
            if not before and not after:
               locations.append([start, end])

    locations.sort(key=cmp)
    i=0
    while i < len(locations)-1:
        if locations[i][1]>locations[i+1][0]:
            if locations[i][0]==locations[i+1][0]:
                if locations[i][1]<locations[i+1][1]:
                    locations.pop(i)
                elif locations[i][1]>locations[i+1][1]:
                    locations.pop(i+1)
            elif locations[i][0]<locations[i+1][0] and locations[i][1]>=locations[i+1][1]:
                locations.pop(i+1)
        else:
            i+=1

    backSymbols = ['.', '/']       
    forwardSymbols = ['.', '#']   
    newLocations = []
    newMethodeName = []

    for location in locations:
        sta = location[0]
        end = location[1]
        ifMethod = False
        packagePath = ''
        if sta>0 and str(message[sta-1]) in backSymbols:
            newSta = sta-1
            while newSta>=0 and str(message[newSta])!=' ':
                packagePath = str(message[newSta])+packagePath
                newSta-=1
            sta = newSta+1

        if end<len(message) and str(message[end]) in forwardSymbols:
            newEnd = end+1
            while newEnd<len(message) and str(message[newEnd])!=' ':
                newEnd+=1
            end = newEnd
            ifMethod = True
        if ifMethod:
            newMethodeName.append([sta, end])
        newLocations.append([sta, end])

        if packagePath != '':
            index = 0
            while index>=0:
                index = message.find(packagePath,index,len(message))
                if index == sta:
                    index = end
                elif index != -1:
                    indexEnd = index+len(packagePath)
                    while indexEnd< len(message) and str(message[indexEnd]) != " ":
                        indexEnd+=1
                    newLocations.append([index,indexEnd])
                    index+=1


    newLocations.sort(key=cmp)
    newMethodeName.sort(key=cmp)
    # replace tokens in message with <file_name>
    end = 0
    new_message = ""
    for location in newLocations:
        start = location[0]
        new_message += message[end:start]
        if location in newMethodeName:
            new_message += " <method_name> "
        else:
            new_message += " <file_name> "
        end = location[1]
    new_message += message[end:len(message)]

    return new_message


def find_file_name2(message,filepaths):

    filePaths = filepaths
    messageOld = message
    message = messageOld.lower()
    replaceTokens = []
    otherMeanWords = ['version','test','assert','junit']
    specialWords = ['changelog','contributing','release','releasenote','readme','releasenotes']
    punctuations = [',', '.', '?', '!', ';', ':', '、']

    for file in filePaths:

        filePathTokens = file.split('/')
        fileName = filePathTokens[-1]

        if fileName.endswith(".md"):
            continue

        if fileName.lower() in message:
            index = message.find(fileName.lower())
            replaceTokens.append(messageOld[index:index+len(fileName)])
        if '.' in fileName:

            newFileName = fileName
            pattern = re.compile(r'(?:\d+(?:\.\w+)+)')
            versions = pattern.findall(newFileName)
            for version in versions:
                if version!=newFileName:
                    newFileName = newFileName.replace(version, '')

            lastIndex = newFileName[1:].rfind('.')
            if lastIndex == -1:
                lastIndex = len(newFileName)-1
            newFileName = newFileName[:lastIndex+1]
            fileNameGreen = newFileName.lower()
       
            if fileNameGreen in specialWords:
                continue
            elif fileNameGreen in otherMeanWords:
                index = 0
                while index!=-1:
                    tempIndex = message[index+1:len(message)].find(fileNameGreen)
                    if tempIndex ==-1:
                        break
                    else:
                        index =index + 1 + tempIndex
                        if index!=-1 and messageOld[index].isupper():
                            replaceTokens.append(messageOld[index:index+len(fileNameGreen)])
                            break

            elif fileNameGreen in message:
                index = message.find(fileNameGreen)
                replaceTokens.append(messageOld[index:index + len(fileNameGreen)])
            else:
              
                fileNameTokens = tokenize(newFileName)
                if len(fileNameTokens) < 2:
                    continue
                if fileNameTokens[0].lower() in message:
                    camelSta = message.find(fileNameTokens[0].lower())
                    camelEnd = -1
                    tempMessag = message[camelSta:]
                    while camelSta >= 0 and len(tempMessag) > 0:
                        tempMessagTokens = tempMessag.split(' ')
                        find = True
                        if tempMessagTokens[0] == fileNameTokens[0].lower():
                           
                            for i in range(0, min(len(tempMessagTokens),len(fileNameTokens))):
                                if len(tempMessagTokens[i])<2:
                                    continue
                                if str(tempMessagTokens[i][-1]) in punctuations:
                                    tempMessagTokens[i] = tempMessagTokens[i][:-1]

                            for i in range(0, len(fileNameTokens)):
                                if i < len(tempMessagTokens) and tempMessagTokens[i] != fileNameTokens[i].lower():
                                    find = False
                                    break
                                elif i > len(tempMessagTokens):
                                    find = False
                                    break
                            if find:
                                lastTokenIndex = tempMessag.find(fileNameTokens[-1].lower())
                                camelEnd = len(tempMessag[:lastTokenIndex]) + len(fileNameTokens[-1])+ camelSta
                                if camelEnd < len(tempMessag) and tempMessag[camelEnd] in punctuations:
                                    camelEnd += 1
                                break
                        index = message[camelSta + 1:].find(fileNameTokens[0].lower())
                        if index == -1:
                            break
                        camelSta = camelSta + 1 + index
                        tempMessag = message[camelSta:]
                    if camelSta!=-1 and camelEnd !=-1:
                        replaceTokens.append(messageOld[camelSta:camelEnd])
    replaceTokens = list(set(replaceTokens))
    return replaceTokens


def find_url(message):
    if 'git-svn-id: ' in message:
    
        pattern = re.compile(
            r'git-svn-id:\s+(?:http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\s+(?:[a-z]|[0-9])+(?:-(?:[a-z]|[0-9])+){4})')
    else:
        pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = re.findall(pattern, message)
    urls = sorted(list(set(urls)), reverse=True)
    for url in urls:
        message = message.replace(url, '<url>')
    return message

def find_IssueLinks(message):
    pattern = re.compile(r'#([0-9]+)|\bgh-([0-9]+)\b', flags=re.I)
    return re.sub(pattern, '<issue_link>',message)

def find_version(message):
    pattern = re.compile(r'[vVr]?\d+(?:\.\w+)+(?:-(?:\w)*){1,2}')
    versions = pattern.findall(message)
    versions = sorted(list(set(versions)),reverse=True)
    for version in versions:
        message = message.replace(version, '<version>')

    pattern2 = re.compile(r'[vVr]?\d+(?:\.\w+)+')
    versions = pattern2.findall(message)

    versions = sorted(list(set(versions)),reverse=True)
    for version in versions:
        message = message.replace(version, '<version>')
    return message

def find_rawCode(message):
    rawCodeSta = message.find('```')
    replaceIden = []
    res = ''
    while rawCodeSta>0:
        rawCodeEnd = message.find('```', rawCodeSta + 3, len(message))
        if rawCodeEnd!=-1:
            replaceIden.append([rawCodeSta,rawCodeEnd+3])
        else:
            break
        rawCodeSta = message.find('```', rawCodeEnd + 3, len(message))
    if len(replaceIden)>0:
        end = 0
        for iden in replaceIden:
            res += message[end:iden[0]]
            end = iden[1]
        res += message[end:len(message)]
        return res
    else:
        return message

def find_SignInfo(message):
    index = message.find("Signed-off-by")
    if index==-1:
        return message
    if index>0 and (message[index-1]=='"' or message[index-1]=="'"):
        return message
    subMessage = message[index:]
    enterIndex = subMessage.find(">")
    message = message[0:index]+" "+message[index+enterIndex+1:]
    return message

def tokenize(identifier):  # camel case splitting
    new_identifier = ""
    identifier = list(identifier)
    new_identifier += identifier[0]
    for i in range(1, len(identifier)):
        if str(identifier[i]).isupper() and (str(identifier[i-1]).islower() or (i < len(identifier)-1 and str(identifier[i+1]).islower())):
            if not new_identifier.endswith(" "):
                new_identifier += " "
        new_identifier += identifier[i]
        if str(identifier[i]).isdigit() and i < len(identifier)-1 and not str(identifier[i+1]).isdigit():
            if not new_identifier.endswith(" "):
                new_identifier += " "
    return new_identifier.split(" ")

def split(path):  # splitting by seperators, i.e. non-alnum
    new_sentence = ''
    for s in path:
        if not str(s).isalnum():
            if len(new_sentence) > 0 and not new_sentence.endswith(' '):
                new_sentence += ' '
            if s != ' ':
                new_sentence += s
                new_sentence += ' '
        else:
            new_sentence += s
    tokens = new_sentence.replace('< enter >', '<enter>').replace('< tab >', '<tab>').\
        replace('< url >', '<url>').replace('< version >', '<version>')\
        .replace('< pr _ link >','<pr_link>').replace('< issue _ link >','<issue_link>')\
        .replace('< otherCommit_link >','<otherCommit_link>').strip().split(' ')
    return tokens


def allennlp_tag(message, predictor):
    result = predictor.predict(message)
    tokens = result['tokens']
    tags = result['pos_tags']

    print("tokens",tokens)
    print("tags",tags)

    indices = []
    for i in range(len(tokens)):
        s = str(tokens[i])
        if s.startswith('file_name>') or s.startswith('version>') or s.startswith('url>') \
                or s.startswith('enter>') or s.startswith('tab>') or s.startswith('iden>') or s.startswith('method_name>')\
                or s.startswith('pr_link>') or s.startswith('issue_link>') or s.startswith('otherCommit_link>'):
            indices.append(i)
        elif s.endswith('<file_name') or s.endswith('<version') or s.endswith('<url') \
                or s.endswith('<enter') or s.endswith('<tab') or s.endswith('<iden') or s.endswith('<method_name')\
                or s.endswith('<pr_link') or s.endswith('<issue_link') or s.endswith('<otherCommit_link'):
            indices.append(i)

    new_tokens = []
    new_tags = []
    for i in range(len(tokens)):
        if i in indices:
            s = str(tokens[i])
            if s.startswith('file_name>'):
                s = s.replace('file_name>', '')
                new_tokens.append('file_name')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('method_name>'):
                s = s.replace('method_name>', '')
                new_tokens.append('method_name')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('version>'):
                s = s.replace('version>', '')
                new_tokens.append('version')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('url>'):
                s = s.replace('url>', '')
                new_tokens.append('url')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('enter>'):
                s = s.replace('enter>', '')
                new_tokens.append('enter')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('tab>'):
                s = s.replace('tab>', '')
                new_tokens.append('tab')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('iden>'):
                s = s.replace('iden>', '')
                new_tokens.append('iden')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('pr_link>'):
                s = s.replace('pr_link>', '')
                new_tokens.append('pr_link')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('issue_link>'):
                s = s.replace('issue_link>', '')
                new_tokens.append('issue_link')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('otherCommit_link>'):
                s = s.replace('otherCommit_link>', '')
                new_tokens.append('otherCommit_link')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.endswith('<file_name'):
                s = s.replace('<file_name', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('file_name')
                new_tags.append('XX')
            elif s.endswith('<method_name'):
                s = s.replace('<method_name', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('method_name')
                new_tags.append('XX')
            elif s.endswith('<version'):
                s = s.replace('<version', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('version')
                new_tags.append('XX')
            elif s.endswith('<url'):
                s = s.replace('<url', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('url')
                new_tags.append('XX')
            elif s.endswith('<enter'):
                s = s.replace('<enter', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('enter')
                new_tags.append('XX')
            elif s.endswith('<tab'):
                s = s.replace('<tab', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('tab')
                new_tags.append('XX')
            elif s.endswith('<iden'):
                s = s.replace('<iden', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('iden')
                new_tags.append('XX')
            elif s.endswith('<pr_link'):
                s = s.replace('<pr_link', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('pr_link')
                new_tags.append('XX')
            elif s.endswith('<issue_link'):
                s = s.replace('<issue_link', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('issue_link')
                new_tags.append('XX')
            elif s.endswith('<otherCommit_link'):
                s = s.replace('<otherCommit_link', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('otherCommit_link')
                new_tags.append('XX')
        else:
            new_tokens.append(tokens[i])
            new_tags.append(tags[i])
    tokens = new_tokens
    tags = new_tags
    length = len(tokens)

    new_tokens = []
    new_tags = []
    targets = ['file_name', 'version', 'url', 'enter', 'tab', 'iden', 'issue_link', 'pr_link', 'otherCommit_link','method_name']
    i = 0
    while i < length:
        if i < length-2 and tokens[i] == '<' and tokens[i+1] in targets and tokens[i+2] == '>':
            new_tokens.append(tokens[i] + tokens[i+1] + tokens[i+2])
            new_tags.append('XX')
            i += 3
        else:
            new_tokens.append(tokens[i])
            new_tags.append(tags[i])
            i += 1

    tokens = new_tokens
    tags = new_tags
    length = len(tokens)
    tokens = ' '.join(tokens)
    tags = ' '.join(tags)
    print('----------------------------------------------------------------------')
    print(tokens)
    print(tags)
    # print(trees)
    return tokens, tags, length

def filter_tokens(length, tokens, tags):
    indices = []
    tokens = tokens.split(' ')
    tags = tags.split(' ')
    for i in range(1, length):
        if str(tokens[i]).startswith('@'):
            indices.append(i)
        elif str(tokens[i]).isalnum() and not str(tokens[i]).islower():
            if str(tags[i]).startswith("NN"):
                # if str(tokens[i]) == 'file_name' or str(tokens[i]) == 'version':
                #     continue
                indices.append(i)
            else:
                before = i>0 and str(tokens[i-1])=="'"
                after = i+1<len(tokens) and str(tokens[i+1]) == "'"
                if before and after:
                    indices.append(i)

    return indices, tokens

def search_in_patches(patches, indices, tokens):
    patches = []
    fount_indices = []
    found_tokens = []
    for index in indices:
        for patch in patches:
            if str(patch).find(tokens[index]) > -1:
                if index>0 and index<len(tokens)-1 and str(tokens[index-1])=="'" and str(tokens[index+1])=="'":
                    found_tokens.append("'" + str(tokens[index]) + "'")
                else:
                    found_tokens.append(tokens[index])
                fount_indices.append(index)
                break

    return fount_indices, list(set(found_tokens))

def get_unreplacable(message, replacement):
    unreplacable_indices = []
    start = 0
    index = str(message).find(replacement, start, len(message))
    while index > -1:
        start = index + len(replacement)
        for i in range(index, start):
            unreplacable_indices.append(i)
        index = str(message).find(replacement, start, len(message))
    return unreplacable_indices


def replace_tokens(message, tokens):
    unreplacable = []
    replacements = ['<file_name>', '<version>', '<url>', '<enter>', '<tab>','<issue_link>', '<pr_link>', '<otherCommit_link>','<method_name>']
    for replacement in replacements:
        unreplacable += get_unreplacable(message, replacement)

    # find out start and end index of replaced tokens
    locations = []
    for t in tokens:
        end = 0
        while end < len(message):
            start = str(message).find(t, end, len(message))
            if start == -1:
                break
            end = start + len(t)
            before = start > 0 and str(message[start - 1]).isalnum()
            after = end < len(message) and str(message[end]).isalnum()
            if not before and not after:
                locations.append([start, end])

    locations.sort(key=cmp)
    i = 0
    while i < len(locations) - 1:
        if locations[i][1] > locations[i + 1][0]:
            if locations[i][0] == locations[i + 1][0]:
                if locations[i][1] < locations[i + 1][1]:
                    locations.pop(i)
                elif locations[i][1] > locations[i + 1][1]:
                    locations.pop(i + 1)
            elif locations[i][0] < locations[i + 1][0] and locations[i][1] >= locations[i + 1][1]:
                locations.pop(i + 1)
        else:
            i += 1

    # merge continuous replaced tokens
    new_locations = []
    i = 0
    start = -1
    while i < len(locations):
        if start < 0:
            start = locations[i][0]
        if i < len(locations) - 1 and locations[i + 1][0] - locations[i][1] < 2:
            i += 1
            continue
        else:
            end = locations[i][1]
            new_locations.append([start, end])
            start = -1
            i += 1

    # replace tokens in message with <file_name>
    end = 0
    new_message = ""
    for location in new_locations:
        start = location[0]
        new_message += message[end:start]
        new_message += "<iden>"
        end = location[1]
    new_message += message[end:len(message)]

    return new_message