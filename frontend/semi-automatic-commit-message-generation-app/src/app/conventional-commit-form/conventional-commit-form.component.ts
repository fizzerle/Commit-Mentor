import {Component, OnInit} from '@angular/core';
import {Commit} from "../model/commit";
import {Type} from "../model/type";
import {StepperSelectionEvent} from "@angular/cdk/stepper";
import * as Diff2Html from "diff2html";
import {Form, FormArray, FormBuilder, FormControl, FormGroup} from "@angular/forms";
import {ApiService} from "../services/api.service";
import {MatStepper} from "@angular/material/stepper";
import {trie, trie_node} from "../util/Trie";
import {MatTreeNestedDataSource} from "@angular/material/tree";
import {NestedTreeControl} from "@angular/cdk/tree";
import {BehaviorSubject, Observable, of as observableOf, throwError} from 'rxjs';
import {DiffFile} from "diff2html/lib/types";
import {catchError} from "rxjs/operators";
import {MatSnackBar} from "@angular/material/snack-bar";

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
export class ConventionalCommitFormComponent implements OnInit {
  commitTypes: string[];
  commitMessage = "";
  private diff!: any;
  loading = false;
  trie = new trie<string>()

  nestedTreeControl: NestedTreeControl<trie_node>;
  nestedDataSource: MatTreeNestedDataSource<trie_node>;
  dataChange: BehaviorSubject<trie_node[]> = new BehaviorSubject<trie_node[]>([]);

  constructor(private fb: FormBuilder, private apiService:ApiService, private snackBar:MatSnackBar) {
    this.commitTypes =Object.values(Type);

    this.userForm = this.fb.group({
      answers: this.fb.array([
      ])
    })

    this.nestedTreeControl = new NestedTreeControl<trie_node>(this._getChildren);
    this.nestedDataSource = new MatTreeNestedDataSource();

    this.dataChange.subscribe(data => this.nestedDataSource.data = data);
    this.init();
  }

  private _getChildren = (node: trie_node) => { return observableOf(node.getChildren()); };

  hasNestedChild = (_: number, nodeData: trie_node) => {return !(nodeData.children.size === 0); };

  ngOnInit(): void {
  }



  model = new Commit();

  submitted = false;

  oldFiles: string[] = []

  onSubmit() { this.submitted = true; }

  selectionChange(stepperEvent: StepperSelectionEvent){
    if(stepperEvent.selectedIndex === 1) {
      let filesToCommit: string[] = []
      for(let node of this.fileNamesOfSelectedLeafNodes(this.trie.root)){
        filesToCommit.push(this.getFullPath(node))
      }
      let missingFiles = this.oldFiles.filter(path => filesToCommit.indexOf(path) < 0);

      console.log(missingFiles)
      if(missingFiles.length > 0){
        console.error("OH NO SOME FILES GOT REMOVED")
        for (let file of missingFiles){
          //does there exits a hunk for that file?
          let indices = [], i;
          for(i = 0; i < this.hunksForQuestions.length; i++){
            if (this.hunksForQuestions[i][0].newName === file){
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
      this.oldFiles = filesToCommit;
      console.log(this.hunksForQuestions)
      this.apiService.filesToCommit(filesToCommit).subscribe(() => {
        if(this.hunksForQuestions.length == 0) this.addQuestion(undefined,false)
      })
      //focus introduces error message that field got change after check see https://github.com/angular/components/issues/12070
      this.focus = false
      this.delay(1000).then(() => {
        this.focus = true
      })
    }
    if(stepperEvent.selectedIndex === 2){
      this.commitMessage = ""
      this.commitMessage += this.getEnumKeyByEnumValue(Type,this.model.type);
      if(this.model.scope) this.commitMessage += "("+ this.model.scope +")"
      this.commitMessage += ": "
      if(this.model.short_description) this.commitMessage += this.model.short_description + "\n"
      else if(this.userForm.value.answers.length > 0 && this.userForm.value.answers[0] != null) this.commitMessage += this.userForm.value.answers[0] + "\n"
      if(this.model.body) this.commitMessage += this.model.body + "\n"

      console.log(this.userForm.value.answers)
      for(let answer of this.userForm.value.answers){
          if(answer != null) this.commitMessage += "* " +answer + "\n"
      }

      if(this.model.breakingChanges) this.commitMessage += "BREAKING CHANGE: "
      if(this.model.closesIssue) {
        this.commitMessage += "Closes "
        this.model.closesIssue.split(",").forEach((id) => this.commitMessage += "#"+id+" ")
      }

    }
  }

  getEnumKeyByEnumValue<T extends {[index:string]:string}>(myEnum: T, enumValue:any) {
    let keys = Object.keys(myEnum).filter(x => myEnum[x] == enumValue);
    return keys.length > 0 ? keys[0] : null;
  }

  outputHtml!: string;
  parsedDiff: DiffFile[] = [];

  init() {

    this.apiService.getGitDiff().subscribe((data) => {
      this.diff = data
      this.parsedDiff = Diff2Html.parse(data, { drawFileList: true, matching: 'lines' });
      let outputHtml = Diff2Html.html(this.parsedDiff, { drawFileList: true, matching: 'lines' });
      this.outputHtml = outputHtml;
      console.log(this.parsedDiff)
      this.parsedDiff.forEach(diffFile => {
        console.log(diffFile.newName)
        this.trie.insert(diffFile.newName,diffFile.newName)
      })
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
  questionsForHunks: string[] = [];
  hunksForQuestions: [DiffFile,string][] = [];
  fileHtml: string = "";
  public focus: boolean = true;

  addQuestion(stepper?: MatStepper,nextFile: boolean = false): void {
    this.loading = true
    this.apiService.getQuestions(nextFile).pipe(catchError(() => {this.loading = false; return throwError("Request had a Error")} )).subscribe((question) => {

      if(question.question == 'Finsih'){
        this.loading = false
        this.snackBar.open("All Questions answered","",{
          duration: 3000
        })
        if(stepper) stepper.next()
        return
      }
      this.questionsForHunks.push(question.question)

      console.log(question)
      console.log(Diff2Html.parse(this.diff))
      var diffFile = Diff2Html.parse(this.diff);
      diffFile = [diffFile[question.file]]
      diffFile[0].blocks = [diffFile[0].blocks[question.hunk]]
      let outputHtml = Diff2Html.html(diffFile,{ drawFileList: false, matching: 'lines',outputFormat: 'side-by-side', });
      this.hunksForQuestions.push([diffFile[0],outputHtml]);
      (this.userForm.get('answers') as FormArray).push(
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
    (this.userForm.get('answers') as FormArray).removeAt(index);
    this.hunksForQuestions.forEach( (item, ind) => {
      if(index === ind) this.hunksForQuestions.splice(index,1);
    });
  }

  getQuestionFormControls(): FormControl[] {
    return (<FormControl[]> (<FormArray> this.userForm.get('answers')).controls)
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
  commitCode() {
    this.committing = true
    console.log("committing ", this.commitMessage)
    this.apiService.postCommit(this.commitMessage).pipe(
      catchError((err) => {
        this.committing = false;
        this.snackBar.open("Request had a Error," + err,"",{
          duration: 2000
        })
        return throwError("Request had a Error" + err)
    }))
      .subscribe(() => {
      this.snackBar.open("Congratulations! Your commit was successful! The page will reload now to prepare for your next commit.","",{
        duration: 2000
      })
      this.delay(3000).then(() => {
        window.location.reload()
      })
    })
  }

  fileClicked(node: trie_node) {
    console.log(node.path)
    console.log(this.trie)
    console.log(node)
    console.log(this.getFullPath(node))
    let diffFile = this.parsedDiff.find(diff => {return diff.newName === this.getFullPath(node)});
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
      console.log(node.path)
      return [node]
    }

    let childrenFiles: trie_node[] = []
    for(const child of node.getChildren()){
      childrenFiles = childrenFiles.concat(this.fileNamesOfSelectedLeafNodes(child))
    }

    return childrenFiles
  }
}
