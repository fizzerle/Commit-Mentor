import {Component, ElementRef, Input, OnInit, ViewChild} from '@angular/core';
import {Commit} from "../model/commit";
import {CommitType} from "../model/CommitType";
import {StepperSelectionEvent} from "@angular/cdk/stepper";
import * as Diff2Html from "diff2html";
import {Form, FormArray, FormBuilder, FormControl, FormGroup} from "@angular/forms";
import {ApiService} from "../services/api.service";
import {MatStepper} from "@angular/material/stepper";
import {trie, trie_node} from "../util/Trie";
import {MatTreeNestedDataSource} from "@angular/material/tree";
import {NestedTreeControl} from "@angular/cdk/tree";
import {BehaviorSubject, interval, Observable, of as observableOf, Subject, throwError} from 'rxjs';
import {DiffFile} from "diff2html/lib/types";
import {catchError, debounce, debounceTime} from "rxjs/operators";
import {MatSnackBar} from "@angular/material/snack-bar";
import {QuestionHunk} from "../model/QuestionHunk";
import {CommitToPublish} from "../model/commitToPublish";
import {Patch} from "../model/Patch";
import {Hunk} from "../model/Hunk";

export class FileNode {
  children!: FileNode[];
  filename!: string;
  type: any;
}

@Component({
  selector: 'app-conventional-commit-form',
  templateUrl: './conventional-commit-form.component.html',
  styleUrls: ['./conventional-commit-form.component.scss']
})
export class ConventionalCommitFormComponent{

  @ViewChild('stepper')
  stepper: MatStepper;

  Type= CommitType
  commitTypes: string[];
  commitPreviewMessage = "";
  diff!: any;
  loading = false;
  trie = new trie<string>()

  nestedTreeControl: NestedTreeControl<trie_node>;
  nestedDataSource: MatTreeNestedDataSource<trie_node>;
  dataChange: BehaviorSubject<trie_node[]> = new BehaviorSubject<trie_node[]>([]);
  openFiles: number = 1;
  openHunks: number = 1;
  allHunksForCurrentFile: number = 1;
  private _projectPath: string ="";
  selectedCommit: Commit = new Commit(0);
  questionsForSelectedCommitType: number[] = [];
  private intervalId: number;

  private motivatingMessages = [" most of the time developing is spent on understanding code changes, so to speed up the understanding describe in detail what and why you changed something",
    "developers search multiple times per month for side effects, maturity stage, selected alternative and constraints however they are seldom recorded",
    "if commit messages are well structured they can be used for automatic changelog creation and triggering of automation's like deployments or running of specific tests",
    "maintenance can typically be expected to use up 70-90% of the overall project budget, documentation like commit messages help to improve maintainability",
    "commit messages can be used for automatic bug localisation "]

  @Input() set projectPath(value: string) {
    this._projectPath = value;
    this.getDiffFromBackend(this._projectPath);
  }

  get projectPath(){
    return this._projectPath
  }

  constructor(private fb: FormBuilder, private apiService:ApiService, private snackBar:MatSnackBar) {
    this.commitTypes =Object.values(CommitType);

    this.userForm = this.fb.group({
      answers: this.fb.array([
      ]),
      belongsTo: this.fb.array([
      ])
    })

    this.nestedTreeControl = new NestedTreeControl<trie_node>(this._getChildren);
    this.nestedDataSource = new MatTreeNestedDataSource();

    this.dataChange.subscribe(data => this.nestedDataSource.data = data);
    this.modelChanged
      .pipe(
        debounceTime(1000))
      .subscribe(() => {
        this.checkMessage()
      })

    this.intervalId = setInterval(() => this.getMotivatingFact(), 30000);
    this.getMotivatingFact()
  }

  private _getChildren = (node: trie_node) => { return observableOf(node.getChildren()); };

  hasNestedChild = (_: number, nodeData: trie_node) => {return !(nodeData.children.size === 0); };

  commits: Commit[] = []
  filesToCommit: string[] = []
  submitted = false;

  oldFiles: string[] = []

  stepperSelectionChanged(stepperEvent: StepperSelectionEvent){
    if(stepperEvent.selectedIndex === 1) {
      this.filesToCommit = []
      for(let node of this.fileNamesOfSelectedLeafNodes(this.trie.root)){
        this.filesToCommit.push(this.getFullPath(node))
      }
      let missingFiles = this.oldFiles.filter(path => this.filesToCommit.indexOf(path) < 0);
      console.log("missing Files", missingFiles)
      if(missingFiles.length > 0){
        console.error("OH NO SOME FILES GOT REMOVED")
        for (let file of missingFiles){
          //does there exits a hunk for that file?
          let indices = [], i;
          for(i = 0; i < this.questionHunks.length; i++){
            let name = this.questionHunks[i].filePath
            if (name === file){
              indices.push(i);
            }
          }
          console.log(indices)
          indices = indices.reverse()
          for(let index of indices){
            this.removeQuestion(index)
          }
        }
      }
      this.oldFiles = this.filesToCommit;
      console.log("Hunks after removal",this.questionHunks)
      this.apiService.filesToCommit(this.filesToCommit).subscribe((openFileNumber) => {
        if(this.questionHunks.length == 0) this.addQuestionHunk(undefined,false)
        else this.openFiles = openFileNumber
      })
      //focus introduces error message that field got change after check see https://github.com/angular/components/issues/12070
      this.focus = false
      this.delay(1000).then(() => {
        this.focus = true
      })
    }
    if(stepperEvent.selectedIndex === 2){
      this.finishMessageSelected()
    }
    if(stepperEvent.selectedIndex === 3) {
      this.apiService.allCommitsComitted().subscribe()
    }
  }

  /*
  Maps the answers and belongs to information to commit objects, after this step all the hunks are assigned to a commit
   */
  private finishMessageSelected() {
    console.log(this.userForm.getRawValue())
    this.commits = []
    this.userForm.getRawValue().answers.forEach((answer: string,index:number) => {
      if (answer !== null) {
        const newCommit = new Commit(this.commits.length,undefined,undefined,answer)
        newCommit.mainHunk = index
        newCommit.hunks.push(index)
        this.commits.push(newCommit)
      } else {
        const indexToWhichCommitHunkBelongs = this.userForm.value.belongsTo[index]
        const commitHunkBelongsTo = this.commits.find((commit) => commit.mainHunk === indexToWhichCommitHunkBelongs);
        if(commitHunkBelongsTo === undefined){
          console.error("Tried to find the commit the hunk "+ index + "belongs to, but could not find it." +
            "This should not occur, because the UI makes sure every hunk is assigned either a message or it belongs to a commit")
        } else{
          commitHunkBelongsTo.hunks.push(index)
        }
      }
    })
    this.selectedCommit = this.commits[0]
    this.buildCommitMessageStringFromCommit(this.selectedCommit)
    //transformAnswersInAMessageText()

  }

  getEnumKeyByEnumValue<T extends {[index:string]:string}>(myEnum: T, enumValue:any) {
    let keys = Object.keys(myEnum).filter(x => myEnum[x] == enumValue);
    return keys.length > 0 ? keys[0] : null;
  }

  outputHtml!: string;
  parsedDiff: DiffFile[] = [];

  getDiffFromBackend(path:string){
    this.diffLoading = true
    this.apiService.getGitDiff(path).pipe(
      catchError((err) => {
        this.committing = false;
        if(err.status === 504) {
          this.snackBar.open("Backend not reachable", "", {
            duration: 2000
          })
        }
        else{
          this.snackBar.open(err.error.detail,"",{
            duration: 2000
          })
        }
        this.diffLoading = false
        return throwError("Request had a Error" + err.detail)
      })).subscribe((data) => {
      console.log(data)
      this.diff = data.diff
      this.diffLoading = false
      if(data.diff === null){
        return
      }
      this.parsedDiff = Diff2Html.parse(data.diff, { drawFileList: true, matching: 'lines' });
      let outputHtml = Diff2Html.html(this.parsedDiff, { drawFileList: true, matching: 'lines' });
      this.outputHtml = outputHtml;
      console.log(this.parsedDiff)
      for (let path of data.files){
        this.trie.insert(path[0],path[0],path[1])
      }
      this.trie.root.path = "Commit All Files"
      this.dataChange.next([this.trie.root])
      // data Nodes have to be set manually, because of https://github.com/angular/components/issues/12170
      this.nestedTreeControl.dataNodes = [this.trie.root]
      this.nestedTreeControl.expandAll()
      //this.checkAllChildren(this.trie.root, true)
    });
  }

  /** Toggle a leaf to-do item selection. Check all the parents to see if they changed */
  leafNodeSelect(node: trie_node): void {
    node.selected = !node.selected
    this.checkAllParentsSelection(node,node.selected);
  }

  /* Checks all the parents when a leaf node is selected/unselected */
  checkAllParentsSelection(node: trie_node, selection: boolean): void {
    if(node.parent){
      node.parent.selected = selection;
      this.checkAllParentsSelection(node.parent, node.selected)
    }
  }

  descendantsPartiallySelected(node: trie_node) : boolean {
    const descendants = this.nestedTreeControl.getDescendants(node);
    const result = descendants.some(child => child.selected);
    return result && !this.descendantsAllSelected(node);
  }

  descendantsAllSelected(node: trie_node) : boolean {
    const descendants = this.nestedTreeControl.getDescendants(node);
    return descendants.every(child => child.selected);
  }

  checkAllChildren(node: trie_node, selection: boolean) {
    node.selected = selection
    for(const child of node.getChildren()){
      this.checkAllChildren(child,selection)
    }
    this.checkAllParentsSelection(node, selection)
  }


  userForm: FormGroup;
  questionHunks: QuestionHunk[] = [];
  fileHtml: string = "";

  public focus: boolean = true;

  addQuestionHunk(stepper?: MatStepper, nextFile: boolean = false): void {
    this.loading = true
    this.apiService.getQuestionHunk(nextFile).pipe(catchError(() => {this.loading = false; return throwError("Request had a Error")} )).subscribe((questionHunk) => {

      if(questionHunk.question == 'Finish'){
        this.loading = false
        this.snackBar.open("All Questions answered","",{
          duration: 3000
        })
        if(stepper) stepper.next()
        return
      }

      this.openFiles = questionHunk.openFiles
      this.openHunks = questionHunk.openHunks
      this.allHunksForCurrentFile = questionHunk.allHunksForThisFile

      console.log("Question Hunk",questionHunk)
      console.log("Parsed diff",Diff2Html.parse(this.diff))
      var diffFile = Diff2Html.parse(this.diff);
      diffFile = [diffFile[questionHunk.fileNumber]]
      diffFile[0].blocks = [diffFile[0].blocks[questionHunk.hunkNumber]]
      let outputHtml = Diff2Html.html(diffFile,{ drawFileList: false, matching: 'lines',outputFormat: 'side-by-side', });
      questionHunk.diff = outputHtml
      this.questionHunks.push(questionHunk);
      (this.userForm.get('answers') as FormArray).push(
        this.fb.control(null)
      );
      (this.userForm.get('belongsTo') as FormArray).push(
        this.fb.control(null)
      );
      this.delay(200).then(() => {
        this.focus = false
      })
      this.loading = false
    });

  }

  async delay(ms: number) {
    return new Promise( resolve => setTimeout(resolve, ms) );
  }

  removeQuestion(index: number) {
    console.log("Removing Question Hunk with index "+index);
    (this.userForm.get('answers') as FormArray).removeAt(index);
    (this.userForm.get('belongsTo') as FormArray).removeAt(index);
    this.questionHunks.forEach( (item, ind) => {
      if(index === ind) this.questionHunks.splice(index,1);
    });
  }

  getQuestionFormControls(): FormControl[] {
    return (<FormControl[]> (<FormArray> this.userForm.get('answers')).controls)
  }

  getBelongsToFormControls(): FormControl[] {
    return (<FormControl[]> (<FormArray> this.userForm.get('belongsTo')).controls)
  }

  send(values: [string]) {
    console.log(values);
  }

  scrollToBottom() {
    window.scrollTo({
      top: document.body.scrollHeight,
      behavior: 'smooth',
    })
  }

  committing = false;
  diffLoading = false;

  checkMessage() {
    let patches: Patch[] = []

    this.selectedCommit.hunks.forEach((hunkNumber) => {
      let questionHunk = this.questionHunks[hunkNumber]
      let found = patches.find((patch) => {patch.patchNumber === questionHunk.fileNumber});
      if(found){
        found.hunks.push(new Hunk(questionHunk.hunkNumber,this.selectedCommit.finalMessage))
      }else {
        patches.push(new Patch(questionHunk.fileNumber,questionHunk.filePath,[new Hunk(questionHunk.hunkNumber,"")],))
      }
    })

    let commitToPublish = new CommitToPublish(this.selectedCommit.finalMessage,patches,this.selectedCommit.id)

    this.apiService.checkMessage(commitToPublish).pipe(
      catchError((err) => {
        this.snackBar.open("Request had a Error," + err,"",{
          duration: 2000
        })
        return throwError("Request had a Error" + err)
      }))
      .subscribe((messageScore) => {
        console.log("Response score was: ", messageScore)
        if(messageScore.body !== null){
          this.calculateMessageStrength(messageScore.body)
        }
      })
  }

  calculateMessageStrength(messageScore: number){
    console.log("messagescore was:", messageScore)
    if(messageScore <= 0.6) this.messageStrength = 0;
    if(messageScore > 0.6 && messageScore <= 0.7) this.messageStrength = 1;
    if(messageScore > 0.7 && messageScore <= 0.8) this.messageStrength = 2;
    if(messageScore > 0.8 && messageScore <= 0.9) this.messageStrength = 3;
    if(messageScore > 0.9 ) this.messageStrength = 4;
  }

  commitCode(form: any) {
    if(!form.valid) return
    this.committing = true
    console.log("committing ", this.commitPreviewMessage)
    let patches: Patch[] = []

    this.selectedCommit.hunks.forEach((hunkNumber) => {
      let questionHunk = this.questionHunks[hunkNumber]
      let found = patches.find((patch) => {patch.patchNumber === questionHunk.fileNumber});
      if(found){
        found.hunks.push(new Hunk(questionHunk.hunkNumber,this.selectedCommit.finalMessage))
      }else {
        patches.push(new Patch(questionHunk.fileNumber,questionHunk.filePath,[new Hunk(questionHunk.hunkNumber,"")],))
      }
    })

    let commitToPublish = new CommitToPublish(this.selectedCommit.finalMessage,patches,this.selectedCommit.id)
    this.apiService.postCommit(commitToPublish).pipe(
      catchError((err) => {
        this.committing = false;
        this.snackBar.open("Request had a Error," + err,"",{
          duration: 2000
        })
        return throwError("Request had a Error" + err)
    }))
      .subscribe(() => {
        this.committing = false;
        this.selectedCommit.commited = true
      this.snackBar.open("Congratulations! Your commit was successful!","",{
        duration: 2000
      })
        if(this.allCommitsCommited()){
          this.delay(2000).then(() => {
            this.snackBar.open("You finished all commits. Please fill out the Diary Study. Thank you :)","",{
              duration: 4000
            })
            this.stepper.selectedIndex = 3
          })
        }
    })
  }

  allCommitsCommited (){
    let numberOfCommitedCommits =  this.commits.filter(commit => commit.commited === true).length
    return numberOfCommitedCommits === this.commits.length
  }

  fileClicked(node: trie_node) {
    console.log(node.path)
    console.log(this.trie)
    console.log(node)
    console.log(this.getFullPath(node))
    let diffFile = this.parsedDiff.find(diff => {
      if(diff.newName === "" || diff.isNew){
        return diff.newName === this.getFullPath(node)
      } else {
        return diff.oldName === this.getFullPath(node)
      }
    });
    if (diffFile){
      this.fileHtml = Diff2Html.html([diffFile],{ drawFileList: false, matching: 'lines' })
    }

  }

  public getFullPath(node: trie_node): string {
    let path = ""
    let currentNode = node
    while (currentNode.parent){
      path = currentNode.path + "/" +path
      currentNode =  currentNode.parent
    }
    return path.slice(0,path.length -1)
  }

  fileNamesOfSelectedLeafNodes(node: trie_node): trie_node[]{
    //console.log(node)
    if(node.terminal && node.selected){
      return [node]
    }

    let childrenFiles: trie_node[] = []
    for(const child of node.getChildren()){
      childrenFiles = childrenFiles.concat(this.fileNamesOfSelectedLeafNodes(child))
    }

    return childrenFiles
  }

  /*
  Builds the final commit message in one string from a commit object
   */
  buildCommitMessageStringFromCommit(commit: Commit) {
    commit.finalMessage = ""
    if(commit.type) commit.finalMessage = ""+this.getEnumKeyByEnumValue(CommitType,commit.type);
    if(commit.scope) commit.finalMessage += "("+ commit.scope +")"
    if(commit.breakingChanges) commit.finalMessage += "!"
    if(commit.type || commit.scope) commit.finalMessage += ": "
    if(commit.short_description) commit.finalMessage += commit.short_description + "\n"

    if(commit.body) commit.finalMessage += "\n" + commit.body + "\n\n"
    console.log(commit.closesIssue)
    if(commit.closesIssue) {
      let ids = ""
      commit.closesIssue.split(",").forEach((id) => {
        id = id.trim()
        const number = Number(id);
        if (!isNaN(number) && number !== 0) ids += "#"+id+" "
        console.log(number)
      })
      if(ids.length > 0)commit.finalMessage += "Closes "+ ids
    }
    this.modelChanged.next();
    this.commitTypeChanged()
  }
  modelChanged = new Subject<string>();

  getTypeLength(modelType: CommitType|undefined): number {
    let desc = this.getEnumKeyByEnumValue(CommitType,modelType)
    if(desc === null) return 0
    return desc.length
  }

  commitDescriptionLength(type: CommitType|undefined, scope: string|undefined, short_description: string|undefined, breaking_changes: boolean|undefined) {
    let typeFull = this.getEnumKeyByEnumValue(CommitType,type)
    let typeLength = 0
    let scopeLength = 0
    let decriptionLength = 0
    let breakingLength =  breaking_changes === undefined || !breaking_changes? 0:1
    if(typeFull !== null) typeLength = typeFull.length
    if(scope !== undefined){
      scopeLength = scope.length + (scope.length === 0? 0:2)
    }
    if(short_description !== undefined) decriptionLength = short_description.length

    return (typeLength + 2) + breakingLength + scopeLength + decriptionLength
  }

  getCircleColorClass(char:any) {
    let color = ""
    if(char === 'M') color = "blue"
    if(char === 'D') color =  "red"
    if(char === 'N') color = "green"
    return "roundCircle " + color
  }

  firstStepSubmit(stepper: MatStepper) {
    if(this.fileNamesOfSelectedLeafNodes(this.trie.root).length === 0){
      this.snackBar.open("Please select minimum one file to commit","",{
        duration: 3000
      })
      return
    }
    stepper.next()
  }

  onCommitSelection(commit: Commit) {
    this.selectedCommit = commit;
    this.commitTypeChanged()
    this.buildCommitMessageStringFromCommit(this.selectedCommit)
    this.checkMessage()
  }

  private defaultColours = [
    'darkred',
    'orangered',
    'orange',
    'yellowgreen',
    'green'
  ];

  messageStrength = 0
  feedback = {} as { suggestions: string[]; warning: string } ;

  getMeterFillColor(strength: number) {
    if (!strength || strength < 0 || strength > 5) {
      return this.defaultColours[0];
    }

    return this.defaultColours[strength]
  }

  questionsForCommitType: {heading: string, questions: string[]}[]  = [
    {
      heading: "Describe Issue",
      questions: ["Where and how did the error occur?",
        "Is the change due to warnings or errors of a tool?",
        "What was the shortcoming of the current solution?"
      ],
    },
    {
      heading: "Illustrate requirement",
      questions: ["Was something out of date?",
        "Why did you need to make this change?",
        "Did the runtime or development environment change?"
      ],
    },
      {
        heading: "Describe objective",
        questions: ["What improvement does your change bring?",
          "How have you fixed the problem?"
        ],
      },
      {
        heading: "Imply necessity",
        questions: ["What functional or non functional (maintainability/readability) improvement does this change bring?",
          "Do you make these changes because of some standard or convention?",
          "Has this commit a relation to a prior commit?",
          "Is this commit part of a larger feature or goal?"
        ],
      },
      {
        heading: "Often neglected",
        questions: ["What were the alternatives considered to the selected approach?",
          "What are the constraints that lead to this approach?",
          "What are the side effects of the approach taken?",
          "How would you describe the code maturity?"
        ],
      }
  ]

  commitTypeQuestionMapping: {[commitype: string] : number[]} = {
    "withoutType":
      [4,3,0,1,2],
    "fix":
      [4,0,3,1,2],
    "feat":
      [4,3,1,0,2],
    "build":
      [4,3,1,0,2],
    "ci":
      [4,3,1,0,2],
    "docs":
      [3,1,2,0],
    "style":
      [3,1,2,0],
    "refactor":
      [4,3,1,2,0],
    "perf":
      [4,3,1,2,0],
    "test":
      [3,1,2,0],
  }

  commitTypeChanged() {
    let selectedType = this.getEnumKeyByEnumValue(CommitType,this.selectedCommit.type)
    if(selectedType !== null){
      this.questionsForSelectedCommitType = this.commitTypeQuestionMapping[selectedType]
    } else{
      this.questionsForSelectedCommitType = []
    }
  }

  public answers: string[] = ["","","","","","","","","","","",""]
  randomMotivation: string;

  saveDiaryEntry() {
    console.log(this.answers)
    this.apiService.saveDiaryEntry(this.answers).subscribe(() => {
      this.delay(2000).then(() => {
        this.snackBar.open("You are finished with all commits. The page will reload now to prepare for your next commits.","",{
          duration: 2000
        })
        this.delay(3000).then(() => {
          window.location.reload()
        })
      })
    })
  }

  getMotivatingFact() {
    this.randomMotivation = this.motivatingMessages[Math.floor(Math.random() * this.motivatingMessages.length)]
  }
}
