<script lang="ts">
	import type { ProblemInfo, Solution } from '$lib/types';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import AdvancedSidebar from './advanced-sidebar.svelte';

	let { explanationsAvailable }: { explanationsAvailable: boolean } = $props();

	const problem = {
		objectives: [{ symbol: 'f_1', name: 'Objective 1', maximize: true, ideal: 2, nadir: 0 }]
	} as ProblemInfo;
	const solutions = [
		{
			state_id: 42,
			solution_index: 0,
			objective_values: { f_1: 1 },
			variable_values: {}
		}
	] as Solution[];

</script>

{#snippet explanationChat()}
	<div data-testid="explanation-chat">Grounded explanation chat</div>
{/snippet}

<Sidebar.Provider>
	<AdvancedSidebar
		{problem}
		preferenceValues={[]}
		{solutions}
		multipliers={explanationsAvailable ? { f_1: 1 } : null}
		tradeoffs={null}
		activeObjectives={['f_1']}
		selectedSolutions={[0]}
		{explanationChat}
	/>
</Sidebar.Provider>
