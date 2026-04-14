from transformers import PreTrainedTokenizerFast, AutoTokenizer
import selfies as sf


TOKENIZER_PATH = "/root/autodl-tmp/nash_merging/MolGen"
test_smi = "[C] [#C] [B] [N] [B] [Branch1] [C] [C] [N] [B] [Branch1] [Ring1] [C] [#C] [N] [Ring1] [=Branch2] [C]"
# test_smi = sf.encoder(test_smi)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
encoding = tokenizer(test_smi)
print("tokens:", tokenizer.convert_ids_to_tokens(encoding['input_ids']))
print("ids:", encoding['input_ids'])