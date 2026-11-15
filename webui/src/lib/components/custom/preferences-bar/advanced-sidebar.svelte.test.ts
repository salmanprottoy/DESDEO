import { render, screen } from '@testing-library/svelte';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import AdvancedSidebarHarness from './advanced-sidebar.test-harness.svelte';

class ResizeObserverMock {
	observe = vi.fn();
	unobserve = vi.fn();
	disconnect = vi.fn();
}

describe('AdvancedSidebar explanation chat', () => {
	beforeAll(() => {
		vi.stubGlobal('ResizeObserver', ResizeObserverMock);
	});

	it('renders the chat only when selected-solution explanations are available', async () => {
		const view = render(AdvancedSidebarHarness, { explanationsAvailable: false });

		expect(screen.queryByTestId('explanation-chat')).toBeNull();

		await view.rerender({ explanationsAvailable: true });

		expect(screen.getByTestId('explanation-chat')).toBeTruthy();
	});
});
