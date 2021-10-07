import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import {Question} from "../model/question";

@Injectable()
export class ApiService {
  constructor(private http: HttpClient) { }

  getGitDiff() {
    return this.http.get<string>("/api/getDiff")
  }

  getQuestions() {
    return this.http.get<Question>("/api/getQuestions")
  }
}
