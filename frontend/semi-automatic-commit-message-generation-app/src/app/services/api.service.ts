import { Injectable } from '@angular/core';
import {HttpClient, HttpErrorResponse} from '@angular/common/http';

import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import {Question} from "../model/question";
import {MatSnackBar} from "@angular/material/snack-bar";

@Injectable()
export class ApiService {
  constructor(private http: HttpClient, private snackBar: MatSnackBar) { }

  getGitDiff() {
    return this.http.get<string>("/api/getDiff")
  }

  getQuestions() {
    return this.http.get<Question>("/api/getQuestions")
  }

  postCommit(commitMessage: string) {
    return this.http.post("/api/commits",{ message: commitMessage },{ observe: 'response' }).pipe(
      catchError(this.handleError)
    );
  }

  private handleError = (error: HttpErrorResponse) => {
    if (error.status === 0) {
      // A client-side or network error occurred. Handle it accordingly.
      console.error('An error occurred:', error.error);
    } else {
      // The backend returned an unsuccessful response code.
      // The response body may contain clues as to what went wrong.
      this.snackBar.open("Something went wrong: See the logs for details","",{
        duration: 3000
      })
      console.error(
        // Simple message.
      `Backend returned code ${error.status}, body was: `, error.error);
    }
    // Return an observable with a user-facing error message.
    return throwError(
      'Something bad happened; please try again later.');
  }
}
