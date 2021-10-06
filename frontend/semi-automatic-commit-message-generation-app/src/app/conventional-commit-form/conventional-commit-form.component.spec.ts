import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConventionalCommitFormComponent } from './conventional-commit-form.component';

describe('ConventionalCommitFormComponent', () => {
  let component: ConventionalCommitFormComponent;
  let fixture: ComponentFixture<ConventionalCommitFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ConventionalCommitFormComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ConventionalCommitFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
