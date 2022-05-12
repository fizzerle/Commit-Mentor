import {Patch} from "./Patch";

export class CommitToPublish {

  constructor(
    public message: string = "",
    public patches: Patch[],
    public id: number
  ) {  }

}
