import {Hunk} from "./Hunk";


export class Patch {

    constructor(
      public patchNumber: number,
      public filename: string,
      public hunks: Hunk[]
    ) {  }

  }
