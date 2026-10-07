import {useEffect,useRef,useState,type ComponentProps} from 'react';
import {TextField} from '@radix-ui/themes';
type Props=Omit<ComponentProps<typeof TextField.Root>,'value'|'onChange'|'type'> & {value:number|string;onValueChange:(value:number)=>void};
/** Keep in-progress text separate from committed numeric settings. */
export default function NumericInput({value,onValueChange,onBlur,onKeyDown,...props}:Props){
 const [draft,setDraft]=useState(String(value));
 const focused=useRef(false),dirty=useRef(false);
 useEffect(()=>{if(!focused.current)setDraft(String(value));},[value]);
 function commit(){const next=Number(draft);if(!draft.trim()||!Number.isFinite(next)){setDraft(String(value));dirty.current=false;return;}setDraft(String(next));if(dirty.current)onValueChange(next);dirty.current=false;}
 return <TextField.Root {...props} type="number" value={draft} onFocus={()=>{focused.current=true;}} onChange={e=>{dirty.current=true;setDraft(e.target.value);}} onBlur={e=>{focused.current=false;commit();onBlur?.(e);}} onKeyDown={e=>{if(!e.nativeEvent.isComposing){if(e.key==='Enter'){e.preventDefault();e.currentTarget.blur();}}onKeyDown?.(e);}}/>;
}
