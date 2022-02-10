import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'semi-automatic-commit-message-generation-app';

  constructor() {
    let path = localStorage.getItem("projectPath")
    if (path != null) {
      this.projectPath = path;
      this.updatedProjectPath = this.projectPath
    }else {
      this.updatedProjectPath = ""
    }
  }
  projectPath: string = "";
  updatedProjectPath: string;

  updateProjectPath() {
    localStorage.setItem("projectPath",this.projectPath);
    this.updatedProjectPath = this.projectPath
  }
}
