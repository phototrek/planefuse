import { test, expect } from '@playwright/test';
import { applyTemplate } from '../src/lib/workspace/template';

const ctx = { stack_name: 'Zion', method: 'pmax', frames: 12, date: '2026-06-16', seq: 1 };

test('applyTemplate: substitutes known tokens', () => {
  expect(applyTemplate('{stack_name}_{method}', ctx)).toBe('Zion_pmax');
  expect(applyTemplate('{stack_name}_{method}_{frames}f_{date}', ctx)).toBe('Zion_pmax_12f_2026-06-16');
});

test('applyTemplate: pads seq to 3 digits', () => {
  expect(applyTemplate('shot_{seq}', ctx)).toBe('shot_001');
});

test('applyTemplate: leaves unknown tokens literal', () => {
  expect(applyTemplate('{stack_name}_{bogus}', ctx)).toBe('Zion_{bogus}');
});

test('applyTemplate: sanitizes filesystem-illegal characters', () => {
  expect(applyTemplate('{stack_name}', { ...ctx, stack_name: 'a/b:c*d' })).toBe('a_b_c_d');
});

test('applyTemplate: missing frames resolves to empty', () => {
  expect(applyTemplate('{stack_name}{frames}', { ...ctx, frames: undefined })).toBe('Zion');
});
