import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GitDiffComponent } from './git-diff.component';

describe('GitDiffComponent', () => {
  let component: GitDiffComponent;
  let fixture: ComponentFixture<GitDiffComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GitDiffComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(GitDiffComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
