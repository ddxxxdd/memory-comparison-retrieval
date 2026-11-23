import pytest
from scripts.run_stale_native_bank_audit import select_native_rows, transition_counts, native_metrics


def row(mid,kind,content='text'):
    return {'memory_id':mid,'scenario_id':'s','content':content,'source_ref':{'dataset':'STALE','record_type':kind}}


def test_filter_exact_base_preserves_all_history_even_if_target_labeled_elsewhere():
    old,history,synthetic=row('old','M_old'),row('h','history'),row('synth','synthetic')
    keep,removed=select_native_rows([old,history,synthetic],[old,history],{'s'})
    assert keep==[old,history] and removed==[synthetic]
    with pytest.raises(AssertionError):
        select_native_rows([old,history,synthetic],[old,row('h','history','changed')],{'s'})


def test_target_entries_exits_and_rank_direction_are_counted_separately():
    before={'q':[str(i) for i in range(1,13)]}
    after={'q':['11','2','3','4','5','6','7','8','9','12','1','10']}
    details={r['memory_id']:r for r in transition_counts(before,after,{'q':['1','2','10','11','12']})}
    assert details['1']['exited_top10'] and details['1']['rank_change']==-10
    assert details['10']['exited_top10']
    assert details['11']['entered_top10'] and details['11']['entered_top3']
    assert details['12']['entered_top10']
    assert details['2']['rank_change']==0


def test_mean_rank_mrr_and_topk_with_single_native_target():
    queries=[{'query_id':'a','source_ref':{'type':'T1'}},{'query_id':'b','source_ref':{'type':'T2'}}]
    rankings={'a':['g','x','y','z'],'b':['x','y','z','g']}
    groups,pq,error=native_metrics(rankings,{'a':'g','b':'g'},queries)
    assert groups['all']['mean_rank']==2.5
    assert groups['all']['mrr']==.625
    assert groups['all']['at_k']['1']['hits']==1
    assert groups['all']['at_k']['3']['hits']==1
    assert groups['all']['at_k']['5']['hits']==2
    assert max(error.values())==0
