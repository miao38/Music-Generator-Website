import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { PianoInfoComponent } from './piano-info.component';

describe('PianoInfoComponent', () => {
  let component: PianoInfoComponent;
  let fixture: ComponentFixture<PianoInfoComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ PianoInfoComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(PianoInfoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
