import {OnInit, ElementRef, Input, Directive, Output, OnChanges, SimpleChanges} from '@angular/core';

@Directive({ selector: '[focusMe]' })
export class FocusDirective implements OnInit, OnChanges  {

  @Input('focusMe') public isFocused: any;

  constructor(private hostElement: ElementRef) { }

  ngOnInit() {
    if (this.isFocused) {
      this.hostElement.nativeElement.focus();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
      this.hostElement.nativeElement.focus();
  }

}
