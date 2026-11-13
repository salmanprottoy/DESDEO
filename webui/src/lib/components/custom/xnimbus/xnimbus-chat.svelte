<script lang="ts">
	import { onMount } from 'svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import {
		chatMethodXnimbusChatPost,
		chatModelsMethodXnimbusChatModelsGet
	} from '$lib/gen/endpoints/DESDEOFastAPI';
	import type {
		AnswerClaim,
		ChatResponse,
		EvidenceFact,
		ModelProfile
	} from '$lib/gen/endpoints/DESDEOFastAPI';

	type ChatStatus = ChatResponse['status'];

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
		verified: { text: 'Server-rendered from verified DESDEO evidence', tone: 'text-green-700' },
		partial: { text: 'Partly answered using verified DESDEO evidence', tone: 'text-amber-700' },
		abstained: { text: 'The available evidence does not answer this', tone: 'text-gray-600' },
		rejected: { text: 'The model answer failed verification and was withheld', tone: 'text-red-700' },
		blocked: { text: 'Use the XNIMBUS controls to change the session', tone: 'text-gray-600' },
		fallback: { text: 'Computed fallback; model output was unavailable or withheld', tone: 'text-amber-700' },
		unavailable: { text: 'No model or computed answer is available', tone: 'text-gray-600' }
	};

	let { problemId, stateId, solutionIndex }: Props = $props();
	let question = $state('');
	let isSending = $state(false);
	let messages = $state<Message[]>([]);
	let error = $state<string | null>(null);
	let conversationId = $state<number | null>(null);
	let models = $state<ModelProfile[]>([]);
	let selectedProfileId = $state<string | null>(null);
	let cloudConsent = $state(false);
	let isLoadingModels = $state(true);

	let selectedProfile = $derived(models.find((model) => model.profile_id === selectedProfileId) ?? null);
	let cloudSelected = $derived(selectedProfile?.deployment === 'cloud');
	let canSend = $derived(
		stateId !== null && !isSending && question.trim().length > 0 && (!cloudSelected || cloudConsent)
	);

	$effect(() => {
		void problemId;
		conversationId = null;
		messages = [];
	});

	onMount(async () => {
		try {
			const response = await chatModelsMethodXnimbusChatModelsGet();
			if (response.status !== 200) throw new Error('Unable to load configured chat models.');
			models = response.data.models;
			selectedProfileId = response.data.default_profile_id ?? null;
		} catch (caught) {
			error = caught instanceof Error ? caught.message : 'Unable to load configured chat models.';
		} finally {
			isLoadingModels = false;
		}
	});

	function formatValue(value: unknown): string {
		if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(6);
		return JSON.stringify(value);
	}

	function describeFact(fact: EvidenceFact): string {
		const { symbol, value, ...rest } = fact.value as { symbol?: string; value?: unknown };
		const index = fact.source.solution_index;
		const where = typeof index === 'number' ? ` (solution ${index + 1})` : '';
		if (symbol !== undefined && value !== undefined) {
			return `${fact.kind.replaceAll('_', ' ')}: ${symbol} = ${formatValue(value)}${where}`;
		}
		const shown = symbol !== undefined ? { symbol, ...rest } : fact.value;
		return `${fact.kind.replaceAll('_', ' ')}: ${JSON.stringify(shown)}${where}`;
	}

	function factsFor(claim: AnswerClaim, response: ChatResponse): EvidenceFact[] {
		return response.evidence.filter((fact) => claim.evidence_ids.includes(fact.evidence_id));
	}

	function errorDetail(data: unknown, fallback: string): string {
		if (typeof data === 'object' && data !== null && 'detail' in data && typeof data.detail === 'string') {
			return data.detail;
		}
		return fallback;
	}

	async function askQuestion(text: string = question) {
		const message = text.trim();
		if (!message || stateId === null || isSending || (cloudSelected && !cloudConsent)) return;

		question = '';
		error = null;
		messages = [...messages, { role: 'user', content: message }];
		isSending = true;

		try {
			const response = await chatMethodXnimbusChatPost({
				problem_id: problemId,
				state_id: stateId,
				solution_index: solutionIndex,
				message,
				conversation_id: conversationId,
				model_profile_id: selectedProfileId,
				cloud_consent: cloudSelected && cloudConsent
			});
			if (response.status !== 200) {
				throw new Error(errorDetail(response.data, `Request failed (${response.status})`));
			}
			const chat = response.data;
			conversationId = chat.conversation_id;
			messages = [...messages, { role: 'assistant', content: chat.answer, response: chat }];
		} catch (caught) {
			error = caught instanceof Error ? caught.message : 'Unable to send the question.';
		} finally {
			isSending = false;
		}
	}

	function resetConversation() {
		conversationId = null;
		messages = [];
		error = null;
	}

	function changeModel(event: Event) {
		const value = (event.currentTarget as HTMLSelectElement).value;
		selectedProfileId = value || null;
		cloudConsent = false;
		resetConversation();
	}
</script>

<section class="rounded-lg border bg-white p-3 shadow-sm" aria-labelledby="xnimbus-chat-title">
	<div class="mb-2 flex items-center justify-between gap-2">
		<h2 id="xnimbus-chat-title" class="text-sm font-semibold">Ask about solution {solutionIndex + 1}</h2>
		<div class="flex items-center gap-2">
			<span class="text-xs text-gray-500">State {stateId ?? '—'}</span>
			{#if messages.length > 0}
				<Button variant="ghost" size="sm" onclick={resetConversation} disabled={isSending}>New chat</Button>
			{/if}
		</div>
	</div>

	<div class="mb-3 space-y-1">
		<label for="xnimbus-chat-model" class="text-xs font-medium text-gray-700">Language model</label>
		<select
			id="xnimbus-chat-model"
			class="w-full rounded border px-2 py-1 text-xs"
			value={selectedProfileId ?? ''}
			disabled={isLoadingModels || isSending}
			onchange={changeModel}
		>
			<option value="">Computed fallback only</option>
			{#each models as model (model.profile_id)}
				<option value={model.profile_id}>{model.label} · {model.deployment}</option>
			{/each}
		</select>
		{#if cloudSelected && selectedProfile}
			<label class="flex items-start gap-2 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
				<input type="checkbox" bind:checked={cloudConsent} class="mt-0.5" />
				<span>
					Allow this chat session to send XNIMBUS evidence to {selectedProfile.provider}. No cloud
					request is made without this consent.
				</span>
			</label>
		{:else if selectedProfile}
			<p class="text-xs text-green-700">Local model; evidence stays on this server.</p>
		{:else if !isLoadingModels}
			<p class="text-xs text-gray-500">No model selected. Answers use the deterministic fallback.</p>
		{/if}
	</div>

	<div class="mb-3 max-h-80 space-y-2 overflow-y-auto" aria-live="polite">
		{#if messages.length === 0}
			<p class="text-xs text-gray-500">
				Ask about objective values, comparisons, active objectives, or reliable trade-off evidence.
				Displayed claims are rendered by the server from computed evidence.
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
						<p class="text-xs text-gray-500">
							State {response.state_id}, solution {response.solution_index + 1}
							{#if response.model_profile}
								· {response.model_profile.label} · {response.model_profile.deployment}
							{/if}
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
										disabled={isSending || stateId === null || (cloudSelected && !cloudConsent)}
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
			placeholder={stateId === null ? 'Select an XNIMBUS solution first.' : 'Ask a question...'}
			rows={2}
			aria-label="Question about the current solution"
		/>
		<Button type="submit" size="sm" disabled={!canSend}>
			{isSending ? 'Checking...' : 'Ask'}
		</Button>
	</form>
</section>
