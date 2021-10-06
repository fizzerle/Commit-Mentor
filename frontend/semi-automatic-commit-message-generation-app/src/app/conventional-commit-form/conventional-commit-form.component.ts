import { Component, OnInit } from '@angular/core';
import {Commit} from "../model/commit";
import {Type} from "../model/type";
import {StepperSelectionEvent} from "@angular/cdk/stepper";

@Component({
  selector: 'app-conventional-commit-form',
  templateUrl: './conventional-commit-form.component.html',
  styleUrls: ['./conventional-commit-form.component.scss']
})
export class ConventionalCommitFormComponent implements OnInit {
  commitTypes: string[];
  commitMessage = "";

  constructor() {
    this.commitTypes =Object.values(Type);
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

}
