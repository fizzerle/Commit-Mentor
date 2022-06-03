import {CommitType} from "./CommitType";
import {QuestionHunk} from "./QuestionHunk";

export class Commit {

    constructor(
      public id: number,
      public type?: CommitType,
      public scope?: string,
      public short_description?: string,
      public body: string = "",
      public breakingChanges?: boolean,
      public footer?: string,
      public closesIssue?: string,
      public hunks: number[] = [],
      public recommendedQuestions: string[] = [],
      public mainHunk?: number,
      public commited = false,
      public finalMessage = ''
    ) {  }

  }
