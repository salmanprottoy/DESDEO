<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';

	type ChatStatus = 'verified' | 'partial' | 'abstained' | 'rejected' | 'blocked' | 'unavailable';

	interface Claim {
		text: string;
		evidence_ids: string[];
	}

	interface Evidence {
		evidence_id: string;
		kind: string;
		value: Record<string, unknown>;
		source: Record<string, string | number>;
	}

	interface ChatResponse {
		answer: string;
		claims: Claim[];
		evidence: Evidence[];
		uncertainty: 'none' | 'partial' | 'insufficient';
		follow_up_questions: string[];
		status: ChatStatus;
		intent: string;
		verified: boolean;
		conversation_id: number;
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

	const STATUS_LABELS: Record<ChatStatus, { text: string; tone: string }> = {
		verified: { text: 'Verified against DESDEO evidence', tone: 'text-green-700' },
		partial: { text: 'Partly answered (only verified claims shown)', tone: 'text-amber-700' },
		abstained: { text: 'The available evidence does not answer this', tone: 'text-gray-600' },
		rejected: { text: 'The model answer failed verification and was withheld', tone: 'text-red-700' },
		blocked: { text: 'Use the XNIMBUS controls to change the session', tone: 'text-gray-600' },
		unavailable: { text: 'The assistant is not configured on this server', tone: 'text-gray-600' }
	};

	let { problemId, stateId, solutionIndex }: Props = $props();
	let question = $state('');
	let isSending = $state(false);
	let messages = $state<Message[]>([]);
	let error = $state<string | null>(null);
	let conversationId = $state<number | null>(null);

	// A conversation belongs to one problem: start a new one when the problem changes.
	$effect(() => {
		void problemId;
		conversationId = null;
		messages = [];
	});

	function formatValue(value: unknown): string {
		if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(6);
		return JSON.stringify(value);
	}

	function describeFact(fact: Evidence): string {
		const { symbol, value, ...rest } = fact.value as { symbol?: string; value?: unknown };
		const index = fact.source.solution_index;
		const where = typeof index === 'number' ? ` (solution ${index + 1})` : '';
		if (symbol !== undefined && value !== undefined) {
			return `${fact.kind.replaceAll('_', ' ')}: ${symbol} = ${formatValue(value)}${where}`;
		}
		const shown = symbol !== undefined ? { symbol, ...rest } : fact.value;
		return `${fact.kind.replaceAll('_', ' ')}: ${JSON.stringify(shown)}${where}`;
	}

	function factsFor(claim: Claim, response: ChatResponse): Evidence[] {
		return response.evidence.filter((fact) => claim.evidence_ids.includes(fact.evidence_id));
	}

	async function askQuestion(text: string = question) {
		const message = text.trim();
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
					message,
					conversation_id: conversationId
				})
			});
			const data = (await response.json()) as ChatResponse | { detail?: string };
			if (!response.ok) {
				const detail = 'detail' in data && typeof data.detail === 'string' ? data.detail : null;
				throw new Error(detail ?? `Request failed (${response.status})`);
			}
			const chat = data as ChatResponse;
			conversationId = chat.conversation_id;
			messages = [...messages, { role: 'assistant', content: chat.answer, response: chat }];
		} catch (caught) {
			error = caught instanceof Error ? caught.message : 'Unable to send the question.';
		} finally {
			isSending = false;
		}
	}

	function newConversation() {
		conversationId = null;
		messages = [];
		error = null;
	}
</script>

<section class="rounded-lg border bg-white p-3 shadow-sm" aria-labelledby="xnimbus-chat-title">
	<div class="mb-2 flex items-center justify-between gap-2">
		<h2 id="xnimbus-chat-title" class="text-sm font-semibold">Ask about solution {solutionIndex + 1}</h2>
		<div class="flex items-center gap-2">
			<span class="text-xs text-gray-500">State {stateId ?? '—'}</span>
			{#if messages.length > 0}
				<Button variant="ghost" size="sm" onclick={newConversation} disabled={isSending}>New chat</Button>
			{/if}
		</div>
	</div>

	<div class="mb-3 max-h-80 space-y-2 overflow-y-auto" aria-live="polite">
		{#if messages.length === 0}
			<p class="text-xs text-gray-500">
				Ask why an objective has its value, compare solutions, or ask about trade-offs. Answers
				are checked against the XNIMBUS results before they are shown.
			</p>
		{:else}
			{#each messages as message, index (index)}
				<div
					class={message.role === 'user'
						? 'ml-6 rounded bg-blue-50 p-2 text-sm'
						: 'mr-6 rounded bg-gray-50 p-2 text-sm'}
				>
					<p>{message.content}</p>
					{#if message.response}
						{@const response = message.response}
						<p class="mt-1 text-xs {STATUS_LABELS[response.status].tone}">
							{STATUS_LABELS[response.status].text}
						</p>
						{#if response.claims.length > 0}
							<details class="mt-1 text-xs">
								<summary class="cursor-pointer text-gray-600">
									Evidence ({response.evidence.length}
									{response.evidence.length === 1 ? 'item' : 'items'})
								</summary>
								<ul class="mt-1 space-y-1">
									{#each response.claims as claim, claimIndex (claimIndex)}
										<li>
											<p class="font-medium">{claim.text}</p>
											<ul class="ml-3 list-disc text-gray-600">
												{#each factsFor(claim, response) as fact (fact.evidence_id)}
													<li title={fact.evidence_id}>{describeFact(fact)}</li>
												{/each}
											</ul>
										</li>
									{/each}
								</ul>
							</details>
						{/if}
						{#if response.follow_up_questions.length > 0}
							<div class="mt-2 flex flex-wrap gap-1">
								{#each response.follow_up_questions as followUp, followUpIndex (followUpIndex)}
									<button
										type="button"
										class="rounded border px-2 py-0.5 text-xs text-gray-700 hover:bg-gray-100"
										disabled={isSending || stateId === null}
										onclick={() => void askQuestion(followUp)}
									>
										{followUp}
									</button>
								{/each}
							</div>
						{/if}
					{/if}
				</div>
			{/each}
		{/if}
	</div>

	{#if error}
		<p class="mb-2 text-xs text-red-600" role="alert">{error}</p>
	{/if}

	<form
		class="space-y-2"
		onsubmit={(event) => {
			event.preventDefault();
			void askQuestion();
		}}
	>
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
