import { Component } from '@angular/core';
import * as Diff2Html from 'diff2html';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'semi-automatic-commit-message-generation-app';

  constructor(private apiService:ApiService, private snackBar:MatSnackBar) {
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
