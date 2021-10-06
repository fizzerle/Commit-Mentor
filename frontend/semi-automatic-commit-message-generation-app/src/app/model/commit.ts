import {Type} from "./type";

export class Commit {

    constructor(
      public type?: Type,
      public scope?: string,
      public short_description?: string,
      public body?: string,
      public breakingChanges?: boolean,
      public footer?: string,
      public closesIssue?: string
    ) {  }
  
  }
