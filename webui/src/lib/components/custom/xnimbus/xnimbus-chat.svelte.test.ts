import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
	chatModels: vi.fn(),
	chat: vi.fn()
}));

vi.mock('$lib/gen/endpoints/DESDEOFastAPI', () => ({
	chatModelsMethodXnimbusChatModelsGet: api.chatModels,
	chatMethodXnimbusChatPost: api.chat
}));

import XNimbusChat from './xnimbus-chat.svelte';

describe('XNimbusChat', () => {
	beforeEach(() => {
		api.chatModels.mockResolvedValue({
			status: 200,
			data: {
				models: [
					{
						profile_id: 'local:qwen-test',
						label: 'Local · qwen-test',
						provider: 'local',
						model_id: 'qwen-test',
						deployment: 'local',
						capabilities: []
					},
					{
						profile_id: 'openai:gpt-test',
						label: 'OpenAI · gpt-test',
						provider: 'openai',
						model_id: 'gpt-test',
						deployment: 'cloud',
						capabilities: []
					}
				],
				default_profile_id: 'local:qwen-test'
			}
		});
		api.chat.mockResolvedValue({
			status: 200,
			data: {
				answer: 'Objective f_1 is 1.0 for solution 3.',
				claims: [],
				evidence: [],
				uncertainty: 'none',
				follow_up_questions: [],
				status: 'verified',
				intent: 'explanation',
				verified: true,
				conversation_id: 5,
				state_id: 42,
				solution_index: 2,
				model_profile: {
					profile_id: 'openai:gpt-test',
					label: 'OpenAI · gpt-test',
					provider: 'openai',
					model_id: 'gpt-test',
					deployment: 'cloud',
					capabilities: []
				}
			}
		});
	});

	it('requires consent and sends the selected solution identity for cloud models', async () => {
		render(XNimbusChat, { problemId: 7, stateId: 42, solutionIndex: 2 });

		const modelSelect = await screen.findByLabelText('Language model');
		await fireEvent.change(modelSelect, { target: { value: 'openai:gpt-test' } });
		const consent = screen.getByRole('checkbox');
		const question = screen.getByLabelText('Question about the current solution');
		await fireEvent.input(question, { target: { value: 'What is f_1?' } });

		expect(screen.getByRole('button', { name: 'Ask' }).disabled).toBe(true);
		await fireEvent.click(consent);
		expect(screen.getByRole('button', { name: 'Ask' }).disabled).toBe(false);
		await fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

		await waitFor(() => {
			expect(api.chat).toHaveBeenCalledWith({
				problem_id: 7,
				state_id: 42,
				solution_index: 2,
				message: 'What is f_1?',
				conversation_id: null,
				model_profile_id: 'openai:gpt-test',
				cloud_consent: true
			});
		});
		expect(await screen.findByText('Objective f_1 is 1.0 for solution 3.')).toBeTruthy();
	});
});
