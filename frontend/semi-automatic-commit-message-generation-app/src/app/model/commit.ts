import {Type} from "./type";
import {QuestionHunk} from "./QuestionHunk";

export class Commit {

    constructor(
      public type?: Type,
      public scope?: string,
      public short_description?: string,
      public body: string = "",
      public breakingChanges?: boolean,
      public footer?: string,
      public closesIssue?: string,
      public hunks: number[] = [],
      public mainHunk?: number,
      public commited = false,
      public finalMessage = ''
    ) {  }

  }
