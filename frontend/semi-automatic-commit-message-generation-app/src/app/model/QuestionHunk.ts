import {DiffFile} from "diff2html/lib/types";

export class QuestionHunk {

  constructor(
    public question: string,
    public fileNumber: number,
    public hunkNumber: number,
    public openFiles: number,
    public openHunks: number,
    public allHunksForThisFile: number,
    public diff: string,
    public filePath: string,
    public diffFile: DiffFile,
  ) {  }

}
