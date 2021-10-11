import {Component, OnInit} from '@angular/core';
import {Commit} from "../model/commit";
import {Type} from "../model/type";
import {StepperSelectionEvent} from "@angular/cdk/stepper";
import * as Diff2Html from "diff2html";
import {FormArray, FormBuilder, FormControl, FormGroup} from "@angular/forms";
import {ApiService} from "../services/api.service";
import {MatStepper} from "@angular/material/stepper";
import {trie, trie_node} from "../util/Trie";
import {MatTreeNestedDataSource} from "@angular/material/tree";
import {NestedTreeControl} from "@angular/cdk/tree";
import {BehaviorSubject, Observable, of as observableOf} from 'rxjs';
import {DiffFile} from "diff2html/lib/types";

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
  private diff!: string;
  loading = false;
  trie = new trie<string>()

  nestedTreeControl: NestedTreeControl<trie_node>;
  nestedDataSource: MatTreeNestedDataSource<trie_node>;
  dataChange: BehaviorSubject<trie_node[]> = new BehaviorSubject<trie_node[]>([]);

  constructor(private fb: FormBuilder, private apiService:ApiService) {
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

  onSubmit() { this.submitted = true; }

  selectionChange(stepperEvent: StepperSelectionEvent){
    if(stepperEvent.selectedIndex === 2){
      this.commitMessage = ""
      this.commitMessage += this.getEnumKeyByEnumValue(Type,this.model.type);
      if(this.model.scope) this.commitMessage += "("+ this.model.scope +")"
      this.commitMessage += ": "
      this.commitMessage += this.model.short_description + "\n"
      if(this.model.body) this.commitMessage += this.model.body + "\n"
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
  private parsedDiff!: DiffFile[];

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
  answeredQuestions: number = 10;
  questionsForHunks: string[] = [];
  hunksForQuestions: string[] = [];
  fileHtml: string = "";

  addQuestion(stepper: MatStepper): void {
    this.loading = true
    this.apiService.getQuestions().subscribe((question) => {
      if(question.question === 'Finish'){
        this.loading = false
        stepper.next()
      }
      this.questionsForHunks.push(question.question)

      var diffFile = Diff2Html.parse(this.diff);
      diffFile = [diffFile[question.file]]
      diffFile[question.file].blocks = [diffFile[question.file].blocks[question.hunk]]
      let outputHtml = Diff2Html.html(diffFile,{ drawFileList: false, matching: 'lines' });
      console.log(diffFile)
      console.log(outputHtml)
      this.hunksForQuestions.push(outputHtml);
      (this.userForm.get('answers') as FormArray).push(
        this.fb.control(null)
      );

      // TODO: edit next line to really refelect how many questions of necessary ones are answered
      this.answeredQuestions += 10
      this.delay(200).then(() => {
        this.scrollToBottom()
      })
      this.loading = false
    });

  }

  async delay(ms: number) {
    return new Promise( resolve => setTimeout(resolve, ms) );
  }

  removeQuestion(index: number) {
    (this.userForm.get('answers') as FormArray).removeAt(index);
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

  commitCode() {
    this.apiService.postCommit(this.commitMessage).subscribe()
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
}
