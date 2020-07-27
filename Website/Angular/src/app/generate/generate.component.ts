import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-generate',
  templateUrl: './generate.component.html',
  styleUrls: ['./generate.component.css']
})
export class GenerateComponent implements OnInit {
  instruments = ["Piano", "Coming Soon"]

  constructor() { }

  ngOnInit(): void {
  }

}
