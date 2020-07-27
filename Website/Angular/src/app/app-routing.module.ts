import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { GenerateComponent } from './generate/generate.component';
import { AboutComponent } from './about/about.component';
import { PianoInfoComponent } from './piano-info/piano-info.component';

const routes: Routes = [
  { path: '', redirectTo: '/generate', pathMatch: 'full' },
  { path: 'generate', component: GenerateComponent },
  { path: 'about', component: AboutComponent },
  { path: 'piano-info', component: PianoInfoComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})

export class AppRoutingModule { }
