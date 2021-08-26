'''
Author WHU ZFJ 2021
CCBERT model definition
'''
import torch
import torch.nn as nn
from torch.nn import _reduction as _Reduction
from torch.nn.functional import nll_loss
import torch.nn.functional as F

ESCAPE_LOG_ZERO = 1e-10
MASK_ATTENTION = 1e10

class LogNllLoss(nn.CrossEntropyLoss):
    '''
    Because the output lever of the decoder is softmax, in order to preserve the gradient,
    the CrosEntropy Loss is removed from the softmax at the beginning 
    '''
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def log_nll(self, input, target, weight=None, size_average=None, ignore_index=-100,
                    reduce=None, reduction='mean'):
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)
        return nll_loss(torch.log(input+ESCAPE_LOG_ZERO), target, weight, None, ignore_index, None, reduction)

    def forward(self, input, target):
        return self.log_nll(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class CopyAttention(nn.Module):
    '''Output the softmax distribution of the vocab_size dimension'''
    def __init__(self, hidden_size, generator, pad_idx):
        super(CopyAttention, self).__init__()
        self.hidden_size = hidden_size
        self.global_attn = nn.Linear(hidden_size*2, hidden_size)
        self.global_attn_v = nn.Linear(hidden_size, 1)
        self.p_gen_linear = nn.Linear(hidden_size*3, 1)
        self.generator = generator # linear from hidden_size to vocab_size
        self.pad_idx = pad_idx
    
    def forward(self, encoder_out, decoder_state, src_ids, tgt_embedding):
        # encoder_out [max_src_len, batch_size, hidden_size]
        # decoder_state [tgt_len(maybe 1), batch_size, hidden_size]
        # src_ids [max_src_len, batch_size]
        # tgt_embedding [tgt_len, batch_size, hidden_size]
        p_gen = None
        # Obtain the semantic vector corresponding to each time step of the decoder ctx [tgt_len, batch_size, hidden_size*2]
        # First get the attention degree of each element output by the encoder
        expanded_decoder_out = decoder_state.unsqueeze(1) # [tgt_len, 1, batch_size, hidden_size]
        # [tgt_len, max_src_len, batch_size, hidden_size]
        expanded_decoder_out = expanded_decoder_out.repeat(1, encoder_out.shape[0], 1, 1)
        expanded_memory = encoder_out.unsqueeze(0) # [1, max_src_len, batch_size, hidden_size]
        # [tgt_len, max_src_len, batch_size, hidden_size]
        expanded_memory = expanded_memory.repeat(decoder_state.shape[0], 1, 1, 1)
        # energy [tgt_len, max_src_len, batch_size, hidden_size]
        energy = torch.tanh(self.global_attn(torch.cat((expanded_decoder_out, expanded_memory), dim = 3)))
        attention = self.global_attn_v(energy).squeeze(3) # [tgt_len, max_src_len, batch_size]
        # src [max_src_len, batch_size] -> [tgt_len, max_src_len, batch_size]
        src = src_ids.unsqueeze(0).repeat(decoder_state.shape[0], 1, 1)
        attn_dist = attention.masked_fill(src==self.pad_idx, -float(MASK_ATTENTION)) # [tgt_len, max_src_len, batch_size]
        attn_prob = F.softmax(attn_dist, dim=1) # [tgt_len, max_src_len, batch_size]
        # Obtain the semantic vector corresponding to each time step of the decoder 
        attn_prob_expanded = attn_prob.transpose(1, 2).unsqueeze(2) # [tgt_len, batch_size, 1, max_src_len]
        attn_prob_expanded = attn_prob_expanded.reshape(-1, 1, encoder_out.shape[0]) # [tgt_len*batch_size, 1, max_src_len]
        expanded_memory = expanded_memory.transpose(1, 2).reshape(-1, encoder_out.shape[0], self.hidden_size) # [tgt_len*batch_size, src_len, hidden_size]
        # ctx [tgt_len, batch_size, hidden_size]
        ctx = torch.bmm(attn_prob_expanded, expanded_memory).squeeze(1).reshape(decoder_state.shape[0], decoder_state.shape[1], self.hidden_size)
        # p_gen_input [tgt_len, batch_size, hidden_size*3]
        p_gen_input = torch.cat((decoder_state, ctx, tgt_embedding), dim=2)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input)) # [tgt_len, batch_size, 1]
        vocab_logits = self.generator(decoder_state)
        vocab_prob = F.softmax(vocab_logits, dim=2) # [tgt_len, batch_size, vocab_size]
        vocab_prob_ = torch.mul(vocab_prob, p_gen)
        attn_prob_ = torch.mul(attn_prob.transpose(1, 2), (1-p_gen)) # [tgt_len, batch_size, src_len]
        final_prob = vocab_prob_.scatter_add(2, src.transpose(1, 2), attn_prob_) # [tgt_len, batch_size, vocab_size]
        return final_prob


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config, beam_size, max_length, sos_id, eos_id, pad_id):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length # max_tgt_len
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.pad_id=pad_id
        self.copy_attention = CopyAttention(config.hidden_size, self.lm_head, pad_id)
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None):   
        '''
        source_ids [batch_size, max_src_len] Start with 0, pad_idx is 1 
        source_mask [batch_size, max_src_len] Composed of 0 and 1, the padding part is 0
        target_ids [batch_size, max_tgt_len] Start with 0, pad_idx is 1 
        target_mask [batch_size, max_tgt_len] Consists of 0 and 1
        '''
        outputs = self.encoder(source_ids, attention_mask=source_mask) # output[0] [batch_size, max_src_len, hidden_dim]
        encoder_output = outputs[0].permute([1,0,2]).contiguous() # [max_src_len, batch_size, hidden_dim]
        if target_ids is not None:  
            # bias [2048, 2048] The upper right triangle is -1e4, and the lower left triangle is 0
            attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]]) # [max_tgt_len, max_tgt_len]
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous() # [max_tgt_len, batch_size, hidden_dim]
            out = self.decoder(tgt_embeddings,encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1-source_mask).bool()) # [max_tgt_len, batch_size, hidden_dim]
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous() # [batch_size, max_tgt_len, hidden_dim]
            # final_prob [tgt_len, batch_size, vocab_size]
            final_prob = self.copy_attention(encoder_output, hidden_states.transpose(0, 1),
                                                source_ids.transpose(0, 1), tgt_embeddings)
            final_prob = final_prob.transpose(0, 1) # [batch_size, tgt_len, vocab_size]
            preds = final_prob.argmax(2) # [batch_size, max_tgt_len]
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1 # Skip the first token
            shift_logits = final_prob[..., :-1, :].contiguous() # Skip the last token output
            shift_labels = target_ids[..., 1:].contiguous() # Skip the first token of the target
            # Flatten the tokens
            loss_fct = LogNllLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
        else:
            #Predict 
            preds=[]           
            zero=torch.cuda.LongTensor(1).fill_(0) # define a tensor valued 0
            for i in range(source_ids.shape[0]): # loop for batch_size times
                context=encoder_output[:,i:i+1] # [max_src_len, 1, hidden_dim]
                context_mask=source_mask[i:i+1,:] # 1, max_src_len, hidden_dim
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState() # [beam_size, tgt_len]
                context=context.repeat(1, self.beam_size,1) # [max_src_len, beam_size, hidden_dim]
                context_mask=context_mask.repeat(self.beam_size,1) # [beam_size, max_src_len, hidden_dim]
                for _ in range(self.max_length): # loop for max_tgt_len times
                    if beam.done():
                        break
                    # The upper right triangle is -1e4, and the lower left triangle is 0 
                    attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]]) # [tgt_len, tgt_len]
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous() # [tgt_len, beam_size, hidden_dim]
                    out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool()) # [tgt_len, beam_size, hidden_dim]
                    out = torch.tanh(self.dense(out))  # [tgt_len, beam_size, hidden_dim]
                    hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:] # [beam_size, hidden_dim]
                    decoder_state = hidden_states.unsqueeze(0) # [1, beam_size, hidden_dim]
                    beam_source_ids = source_ids[i].unsqueeze(0) # [1, max_src_len]
                    beam_source_ids = beam_source_ids.repeat(self.beam_size, 1) # [beam_size, max_src_len]
                    # final_prob [1, beam_size, vocab_size]
                    final_prob = self.copy_attention(context, decoder_state,
                                                beam_source_ids.transpose(0, 1), tgt_embeddings[-1].unsqueeze(0))
                    final_prob = final_prob.squeeze(0) # [beam_size, vocab_size]
                    out = torch.log(final_prob).data # [beam_size, vocab_size]
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin())) # [beam_size, tgt_len]
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1) # [beam_size, tgt_len+1]
                hyp= beam.getHyp(beam.getFinal()) # [beam_size, max_tgt_len]
                pred=beam.buildTargetTokens(hyp)[:self.beam_size] # [beam_size, max_tgt_len]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred] # [beam_size, 1, max_tgt_len]
                # torch.cat(pred,0).unsqueeze(0) [1, beam_size, max_tgt_len]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds = torch.cat(preds,0) # [batch_size, beam_size, max_tgt_len]
            preds = preds[:, 0, :] # [batch_size, max_tgt_len]
            loss = None
        return loss, preds


class Beam(object):
    '''
    originated in
    https://github.com/guoday/CodeBERT/tree/master/CodeBERT/code2nl
    '''
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
