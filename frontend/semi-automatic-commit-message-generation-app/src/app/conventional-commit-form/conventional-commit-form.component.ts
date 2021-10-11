import {Component, OnInit} from '@angular/core';
import {Commit} from "../model/commit";
import {Type} from "../model/type";
import {StepperSelectionEvent} from "@angular/cdk/stepper";
import * as Diff2Html from "diff2html";
import {FormArray, FormBuilder, FormControl, FormGroup} from "@angular/forms";
import {ApiService} from "../services/api.service";

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

  constructor(private fb: FormBuilder, private apiService:ApiService) {
    this.commitTypes =Object.values(Type);

    this.userForm = this.fb.group({
      answers: this.fb.array([
      ])
    })
  }

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

  init() {
    this.apiService.getGitDiff().subscribe((data) => {
      this.diff = data
      let outputHtml = Diff2Html.html(data, { drawFileList: true, matching: 'lines' });
      this.outputHtml = outputHtml;
    });
  }


  userForm: FormGroup;
  answeredQuestions: number = 10;
  questionsForHunks: string[] = [];
  hunksForQuestions: string[] = [];

  addQuestion(): void {
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
}
