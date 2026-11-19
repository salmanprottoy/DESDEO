import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
	chatModels: vi.fn(),
	chat: vi.fn()
}));

// GUI_XNIMBUS talks to the API through the openapi-fetch client in $lib/api/client.
vi.mock('$lib/api/client', () => ({
	api: {
		GET: (path: string) => {
			if (path !== '/method/xnimbus/chat/models') throw new Error(`unexpected GET ${path}`);
			return api.chatModels();
		},
		POST: (path: string, init: { body: unknown }) => {
			if (path !== '/method/xnimbus/chat') throw new Error(`unexpected POST ${path}`);
			return api.chat(init.body);
		}
	}
}));

const ok = (data: unknown) => ({ data, error: undefined, response: { status: 200 } });

import XNimbusChat from './xnimbus-chat.svelte';

describe('XNimbusChat', () => {
	beforeEach(() => {
		api.chatModels.mockResolvedValue(
			ok({
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
			})
		);
		api.chat.mockResolvedValue(
			ok({
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
			})
		);
	});

	it('requires consent and sends the selected solution identity for cloud models', async () => {
		render(XNimbusChat, { problemId: 7, stateId: 42, solutionIndex: 2 });

		const modelSelect = await screen.findByLabelText('Language model');
		await fireEvent.change(modelSelect, { target: { value: 'openai:gpt-test' } });
		const consent = screen.getByRole('checkbox');
		const question = screen.getByLabelText('Question about the current solution');
		await fireEvent.input(question, { target: { value: 'What is f_1?' } });

		const askButton = screen.getByRole<HTMLButtonElement>('button', { name: 'Ask' });
		expect(askButton.disabled).toBe(true);
		await fireEvent.click(consent);
		expect(askButton.disabled).toBe(false);
		await fireEvent.click(askButton);

		await waitFor(() => {
			expect(api.chat).toHaveBeenCalledWith({
				problem_id: 7,
				state_id: 42,
				solution_index: 2,
				message: 'What is f_1?',
				conversation_id: null,
				model_profile_id: 'openai:gpt-test',
				use_model: true,
				cloud_consent: true
			});
		});
		expect(await screen.findByText('Objective f_1 is 1.0 for solution 3.')).toBeTruthy();
	});

	it('shows the API error detail when the chat request fails', async () => {
		api.chat.mockResolvedValue({
			data: undefined,
			error: { detail: 'XNIMBUS state not found.' },
			response: { status: 404 }
		});
		render(XNimbusChat, { problemId: 7, stateId: 42, solutionIndex: 2 });

		const modelSelect = await screen.findByLabelText('Language model');
		await fireEvent.change(modelSelect, { target: { value: '' } });
		const question = screen.getByLabelText('Question about the current solution');
		await fireEvent.input(question, { target: { value: 'What is f_1?' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

		expect(await screen.findByText('XNIMBUS state not found.')).toBeTruthy();
	});

	it('requests deterministic fallback without resolving the default model', async () => {
		render(XNimbusChat, { problemId: 7, stateId: 42, solutionIndex: 2 });

		const modelSelect = await screen.findByLabelText('Language model');
		await fireEvent.change(modelSelect, { target: { value: '' } });
		const question = screen.getByLabelText('Question about the current solution');
		await fireEvent.input(question, { target: { value: 'What is f_1?' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

		await waitFor(() => {
			expect(api.chat).toHaveBeenCalledWith({
				problem_id: 7,
				state_id: 42,
				solution_index: 2,
				message: 'What is f_1?',
				conversation_id: null,
				model_profile_id: null,
				use_model: false,
				cloud_consent: false
			});
		});
	});
	it('sends the question with Enter but not with Shift+Enter', async () => {
		render(XNimbusChat, { problemId: 7, stateId: 42, solutionIndex: 2 });

		const modelSelect = await screen.findByLabelText('Language model');
		await fireEvent.change(modelSelect, { target: { value: '' } });
		const question = screen.getByLabelText('Question about the current solution');
		await fireEvent.input(question, { target: { value: 'What is f_1?' } });

		await fireEvent.keyDown(question, { key: 'Enter', shiftKey: true });
		expect(api.chat).not.toHaveBeenCalled();

		await fireEvent.keyDown(question, { key: 'Enter' });
		await waitFor(() => {
			expect(api.chat).toHaveBeenCalledWith(expect.objectContaining({ message: 'What is f_1?' }));
		});
	});
});
