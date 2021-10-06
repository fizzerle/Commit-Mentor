import { Component, OnInit } from '@angular/core';
import {Commit} from "../model/commit";
import {Type} from "../model/type";

@Component({
  selector: 'app-conventional-commit-form',
  templateUrl: './conventional-commit-form.component.html',
  styleUrls: ['./conventional-commit-form.component.scss']
})
export class ConventionalCommitFormComponent implements OnInit {
  commitTypes: string[];

  constructor() {
    this.commitTypes =Object.values(Type);
  }

  ngOnInit(): void {
  }

  model = new Commit();

  submitted = false;

  onSubmit() { this.submitted = true; }

}
