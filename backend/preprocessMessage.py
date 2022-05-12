import re

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
        elif str(tokens[i]).isalnum():
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
    fount_indices = []
    found_tokens = []
    for index in indices:
        for patch in patches:
            print(str(patch))
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

def replace_newLine(message):
    return message.replace("\n","<enter>")

def preprocessMessageForModel(message,patches,filepaths, predictor):
    message = replace_newLine(message)
    message = find_url(message)
    message = find_version(message)
    message = find_rawCode(message)
    message = find_SignInfo(message)
    message = find_IssueLinks(message)

    if message.strip(" ") == "":
        message = "empty log message"
    message = replace_file_name(message,filepaths)

    tokens, tags, length = allennlp_tag(message, predictor)
    print(tokens,tags,length)
    indices, tokens = filter_tokens(length, tokens, tags)
    print(indices,tokens)
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