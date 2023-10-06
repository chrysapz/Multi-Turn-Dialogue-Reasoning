

    model.eval()

    DEV_BATCH_SIZE = 1
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=DEV_BATCH_SIZE, collate_fn=dev_collate_fn)
    # Calculate perplexity
    generated_info = {'sentence_id':[], 'generated_ids':[], 'perplexity':[]} #(sentence_id, generated_ids, perplexity_of_generated)
    with torch.no_grad():
        for batch in dev_loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key not in ['sentence_id']} #sentence_id is useful only for metrics
            # inputs.pop('labels')
            # note that the difference between input_ids and labels is that in labels we have -100 in ignore tokens
            outputs_ids = model.generate( #! maybe add trainer.model?
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True,
                temperature = 1
            )

            whole_sequences_ids = outputs_ids['sequences'] #(batch_size, input_length+max_new_tokens)
            generated_scores = outputs_ids['scores'] #it's a tuple of len max_new_tokens where each (batch_size, vocab_size)
            
            output_text = tokenizer.decode(whole_sequences_ids[0], skip_special_tokens=True)

            #! not correct
            # Calculate cross-entropy loss for each sequence in the batch
            for i in range(len(whole_sequences_ids)):
                # Get the logits for the generated sequence
                generated_logits = generated_scores[i]

                # Calculate the cross-entropy loss
                loss = torch.nn.functional.cross_entropy(generated_logits, inputs['labels'][i])
                perplexity = torch.exp(loss).item()            
                # Print or store the loss for this sequence
                print(f"Loss for sequence {i}: {loss.item()}")

            if DEV_BATCH_SIZE == 1:
                generated_info['sentence_id'].append(batch['sentence_id'][0])
                generated_info['generated_ids'].append(outputs)
                generated_info['perplexity'].append(perplexity)
            else:
                raise ValueError('We do not support batch size > 1')
