import { normalize, problem, removeRow, setBoundary, splitAt, type Row } from "./timelineEdit";

const ROWS: Row[] = [["place_1", 0, 10], ["torque_1", 10, 20]];

test("a key press at t starts that step there, until the next change", () => {
  expect(splitAt(ROWS, 4, "idle")).toEqual([["place_1", 0, 4], ["idle", 4, 10], ["torque_1", 10, 20]]);
  expect(splitAt(ROWS, 10, "place_1")).toEqual([["place_1", 0, 20]]);  // merged with the same step before it
  expect(splitAt([["idle", 0, 20]], 0, "place_1")).toEqual([["place_1", 0, 20]]);
});

test("boundaries move within their rows and removing a row gives its time to the one before", () => {
  expect(setBoundary(ROWS, 0, 12)).toEqual([["place_1", 0, 12], ["torque_1", 12, 20]]);
  expect(setBoundary(ROWS, 0, 25)[0][2]).toBe(19.95);
  expect(removeRow(ROWS, 1)).toEqual([["place_1", 0, 20]]);
  expect(removeRow(ROWS, 0)).toEqual([["torque_1", 0, 20]]);
});

test("problems are what the server would refuse", () => {
  expect(problem(ROWS, 20)).toBeNull();
  expect(problem(ROWS, 30)).toMatch(/ends at 20/);
  expect(problem(normalize([["a", 0, 5], ["b", 5, 5]]), 5)).toBeNull();
});
