import { Component, OnInit } from '@angular/core';
import {Commit} from "../model/commit";
import {Type} from "../model/type";
import {StepperSelectionEvent} from "@angular/cdk/stepper";
import * as Diff2Html from "diff2html";
import {AbstractControl, FormArray, FormBuilder, FormControl, FormGroup} from "@angular/forms";
import {ApiService} from "../services/api.service";

@Component({
  selector: 'app-conventional-commit-form',
  templateUrl: './conventional-commit-form.component.html',
  styleUrls: ['./conventional-commit-form.component.scss']
})
export class ConventionalCommitFormComponent implements OnInit {
  commitTypes: string[];
  commitMessage = "";
  answer = "";

  constructor(private fb: FormBuilder, private apiService:ApiService) {
    this.commitTypes =Object.values(Type);
    this.init();

    this.userForm = this.fb.group({
      answers: this.fb.array([
        this.fb.control(null)
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
      let outputHtml = Diff2Html.html(data, { drawFileList: false, matching: 'lines' });
      this.outputHtml = outputHtml;
    });
  }


  userForm: FormGroup;
  answeredQuestions: number = 10;

  addQuestion(): void {
    (this.userForm.get('answers') as FormArray).push(
      this.fb.control(null)
    );
    this.answeredQuestions += 10
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

}
