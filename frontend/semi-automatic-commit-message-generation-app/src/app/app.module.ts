import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { GitDiffComponent } from './git-diff/git-diff.component';
import { MatToolbarModule } from '@angular/material/toolbar';
import { ConventionalCommitFormComponent } from './conventional-commit-form/conventional-commit-form.component';
import {FormsModule, ReactiveFormsModule} from "@angular/forms";
import {MatCardModule} from "@angular/material/card";
import {MatGridList, MatGridListModule} from "@angular/material/grid-list";
import {MatCheckboxModule} from "@angular/material/checkbox";
import {MatFormFieldModule} from "@angular/material/form-field";
import {MatSelectModule} from "@angular/material/select";
import {MatInputModule} from "@angular/material/input";
import {MatOptionModule} from "@angular/material/core";
import {MatStepper, MatStepperModule} from "@angular/material/stepper";
import {MatButtonModule} from "@angular/material/button";
import {MatExpansionModule} from "@angular/material/expansion";
import {MatProgressBarModule} from "@angular/material/progress-bar";
import {HttpClientModule} from "@angular/common/http";
import {ApiService} from "./services/api.service";
import {MatSnackBarModule} from "@angular/material/snack-bar";
import {MatTreeModule} from "@angular/material/tree";
import {MatIconModule} from "@angular/material/icon";
import {FocusDirective} from "./conventional-commit-form/focus.directive";
import {MatChipsModule} from "@angular/material/chips";
import {MatProgressSpinnerModule} from "@angular/material/progress-spinner";
import {MatRadioButton, MatRadioModule} from "@angular/material/radio";

@NgModule({
  declarations: [
    AppComponent,
    GitDiffComponent,
    ConventionalCommitFormComponent,
    FocusDirective
  ],
  imports: [
    BrowserModule,
    MatChipsModule,
    MatProgressSpinnerModule,
    AppRoutingModule,
    HttpClientModule,
    MatRadioModule,
    BrowserAnimationsModule,
    MatToolbarModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatCheckboxModule,
    MatGridListModule,
    MatFormFieldModule,
    MatSelectModule,
    MatInputModule,
    MatOptionModule,
    MatStepperModule,
    MatButtonModule,
    MatExpansionModule,
    MatProgressBarModule,
    MatSnackBarModule,
    MatTreeModule,
    MatIconModule
  ],
  providers: [ApiService],
  bootstrap: [AppComponent]
})
export class AppModule { }
