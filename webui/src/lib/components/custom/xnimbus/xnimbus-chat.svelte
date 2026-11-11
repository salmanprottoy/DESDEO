<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';

	interface Claim {
		text: string;
		evidence_ids: string[];
	}

	interface Evidence {
		evidence_id: string;
		kind: string;
		value: Record<string, unknown>;
	}

	interface ChatResponse {
		answer: string;
		claims: Claim[];
		evidence: Evidence[];
		uncertainty: 'none' | 'partial' | 'insufficient';
		follow_up_questions: string[];
		verified: boolean;
	}

	interface Message {
		role: 'user' | 'assistant';
		content: string;
		response?: ChatResponse;
	}

	interface Props {
		problemId: number;
		stateId: number | null;
		solutionIndex: number;
	}

	let { problemId, stateId, solutionIndex }: Props = $props();
	let question = $state('');
	let isSending = $state(false);
	let messages = $state<Message[]>([]);
	let error = $state<string | null>(null);

	async function askQuestion() {
		const message = question.trim();
		if (!message || stateId === null || isSending) return;

		question = '';
		error = null;
		messages = [...messages, { role: 'user', content: message }];
		isSending = true;

		try {
			const response = await fetch('/api/method/xnimbus/chat', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					problem_id: problemId,
					state_id: stateId,
					solution_index: solutionIndex,
					message
				})
			});
			const data = (await response.json()) as ChatResponse | { detail?: string };
			if (!response.ok) {
				throw new Error('detail' in data && data.detail ? data.detail : `Request failed (${response.status})`);
			}
			messages = [...messages, { role: 'assistant', content: (data as ChatResponse).answer, response: data as ChatResponse }];
		} catch (caught) {
			error = caught instanceof Error ? caught.message : 'Unable to send the question.';
		} finally {
			isSending = false;
		}
	}
</script>

<section class="rounded-lg border bg-white p-3 shadow-sm" aria-labelledby="xnimbus-chat-title">
	<div class="mb-2 flex items-center justify-between">
		<h2 id="xnimbus-chat-title" class="text-sm font-semibold">Ask about this solution</h2>
		<span class="text-xs text-gray-500">State {stateId ?? '—'}</span>
	</div>

	<div class="mb-3 max-h-56 space-y-2 overflow-y-auto" aria-live="polite">
		{#if messages.length === 0}
			<p class="text-xs text-gray-500">
				Ask why an objective changed, compare solutions, or request a trade-off explanation.
			</p>
		{:else}
			{#each messages as message}
				<div class={message.role === 'user' ? 'ml-6 rounded bg-blue-50 p-2 text-sm' : 'mr-6 rounded bg-gray-50 p-2 text-sm'}>
					<p>{message.content}</p>
					{#if message.response}
						<p class="mt-1 text-xs text-gray-500">
							{message.response.verified ? 'Verified against DESDEO evidence' : 'Insufficient evidence'}
							{#if message.response.evidence.length > 0}
								· {message.response.evidence.length} evidence items
							{/if}
						</p>
					{/if}
				</div>
			{/each}
		{/if}
	</div>

	{#if error}
		<p class="mb-2 text-xs text-red-600" role="alert">{error}</p>
	{/if}

	<form class="space-y-2" onsubmit={(event) => { event.preventDefault(); askQuestion(); }}>
		<Textarea
			bind:value={question}
			disabled={stateId === null || isSending}
			placeholder={stateId === null ? 'Initialize XNIMBUS first.' : 'Ask a question...'}
			rows={2}
			aria-label="Question about the current solution"
		/>
		<Button type="submit" size="sm" disabled={stateId === null || !question.trim() || isSending}>
			{isSending ? 'Checking...' : 'Ask'}
		</Button>
	</form>
</section>
