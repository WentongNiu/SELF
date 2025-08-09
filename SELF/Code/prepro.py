from tqdm import tqdm

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        is_entity = []
        input_format = self.args.input_format

        entity_label = []
        My_entity_label = []
        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)
            # TODO: entity label
            origin_tokens_wordpiece_len = len(tokens_wordpiece)
            if (ss <= i_t <= se) or (os <= i_t <= oe):
                entity_index = [1.0] * origin_tokens_wordpiece_len
            else:
                entity_index = [0] * origin_tokens_wordpiece_len

            if input_format == 'entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + tokens_wordpiece
                    entity_index = [1] + entity_index
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                    entity_index = entity_index + [1]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['#'] + tokens_wordpiece
                    entity_index = [1] + entity_index
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['#']
                    entity_index = entity_index + [1]
            else:
                raise ValueError(f"invalid input format: {input_format}")

            sents.extend(tokens_wordpiece)

            if (se < i_t < os) or (oe < i_t < ss):
                My_entity_label.extend([1.0] * len(tokens_wordpiece))
            else:
                My_entity_label.extend([0] * len(tokens_wordpiece))

            # Assign entity mask
            if ss <= i_t <= se or os <= i_t <= oe:
                entity_mask = [1 for _ in range(len(tokens_wordpiece))]
                if i_t == ss or i_t == os:
                    entity_mask[0] = 0
                if i_t == se or i_t == oe:
                    entity_mask[-1] = 0
            else:
                entity_mask = [0 for _ in range(len(tokens_wordpiece))]
            is_entity.extend(entity_mask)

            if input_format == 'entity_marker_punct':
                entity_label.extend(entity_index)

        sents = sents[:self.args.max_seq_length - 2]

        entity_label = entity_label[:self.args.max_seq_length - 2]

        is_entity = is_entity[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        return input_ids, new_ss + 1, new_os + 1, [0] + is_entity, [1] + entity_label + [1]


class TACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        # NA must be 0
        self.LABEL_TO_ID = {"NA": 0, "org:subsidiaries": 1, "per:date_of_birth": 2, "per:cause_of_death": 3,
                            "per:age": 4, "per:stateorprovince_of_birth": 5, "per:countries_of_residence": 6,
                            "per:country_of_birth": 7, "per:stateorprovinces_of_residence": 8, "org:website": 9,
                            "per:cities_of_residence": 10, "per:parents": 11, "per:employee_of": 12, "org:founded": 13,
                            "per:city_of_birth": 14, "org:parents": 15, "org:political/religious_affiliation": 16,
                            "per:schools_attended": 17, "per:country_of_death": 18, "per:children": 19,
                            "org:top_members/employees": 20, "per:date_of_death": 21, "org:members": 22,
                            "org:alternate_names": 23, "per:religion": 24, "org:member_of": 25,
                            "org:city_of_headquarters": 26, "per:origin": 27, "org:shareholders": 28,
                            "per:charges": 29, "per:title": 30, "org:number_of_employees/members": 31,
                            "org:dissolved": 32, "org:country_of_headquarters": 33, "per:alternate_names": 34,
                            "per:siblings": 35, "org:stateorprovince_of_headquarters": 36, "per:spouse": 37,
                            "per:other_family": 38, "per:city_of_death": 39, "per:stateorprovince_of_death": 40,
                            "org:founded_by": 41}

    def read(self, file_in):
        features = []
        with open(file_in, 'r', encoding='utf-8') as fr:
            for line in tqdm(fr):
                line = line.rstrip()
                line = eval(line)
                ss, se = line['h']['pos'][0], line['h']['pos'][-1] - 1
                os, oe = line['t']['pos'][0], line['t']['pos'][-1] - 1
                tokens = line['token']
                tokens = [convert_token(token) for token in tokens]
                input_ids, new_ss, new_os, entity_mask, entity_label = self.tokenize(tokens, None, None, ss, se, os, oe)
                rel = self.LABEL_TO_ID[line['relation']]
                feature = {
                    'input_ids': input_ids,
                    'labels': rel,
                    'ss': new_ss,
                    'os': new_os,
                    'entity_mask': entity_mask,
                    'entity_label': entity_label,
                }
                features.append(feature)
        return features




